import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation.phase5_targeted_interventions import evaluate_model_metrics, load_or_create_multilingual_vocabs
from model.ner_model_v4_multilingual import NERMultilingualModel
from training.class_weighted_loss import ClassWeightedCRFLoss
from training.multilingual_dataset import MultilingualNERDataset, collate_fn_multilingual
from training.assamese_strengthening_sweep import get_scaled_class_weights, evaluate_sentence_200_emissions
from weak_supervision.estimate_silver_precision import load_conll_data
from utils.seed import set_global_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proc_dir = PROJECT_ROOT / "data" / "processed"
gold_dir = PROJECT_ROOT / "data" / "gold" / "assamese"
ckpts_dir = PROJECT_ROOT / "checkpoints"

word_to_id, char_to_id = load_or_create_multilingual_vocabs(proc_dir)
dev_path = gold_dir / "dev.conll"
stress_path = gold_dir / "long_multientity_eval_v2.conll"
s200_toks, s200_gold = load_conll_data(stress_path)[0]

def test_epoch_trace(name: str, train_path: Path, multiplier: float):
    print(f"\n=======================================================")
    print(f"Tracing Epochs for {name} (multiplier={multiplier}x)")
    print(f"=======================================================")
    set_global_seed(42)
    train_paths = [train_path, train_path]
    train_ds = MultilingualNERDataset(
        lang_conll_paths={"assamese": train_paths},
        word_to_id=word_to_id,
        char_to_id=char_to_id,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn_multilingual)

    model = NERMultilingualModel(
        word_vocab_size=len(word_to_id), char_vocab_size=len(char_to_id), num_tags=9, num_languages=3,
        word_embedding_dim=100, char_embedding_dim=50, lang_embedding_dim=16,
        filter_widths=(3, 4, 5), filters_per_width=30, hidden_dim=128, num_layers=2,
        num_heads=4, num_attention_blocks=1, attention_dropout=0.1, crf_config="shared",
    ).to(device)

    ckpt = torch.load(ckpts_dir / "improved_assamese_v2.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    weights = get_scaled_class_weights(multiplier)
    loss_fn = ClassWeightedCRFLoss(class_weights=weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    for ep in range(1, 6):
        model.train()
        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()

            batch_size_cur, seq_len_cur = word_ids.shape
            word_repr = model.word_embedding(word_ids)
            lang_repr = model.lang_embedding(lang_ids, seq_len_cur)
            char_repr = model.char_cnn(char_ids)
            combined = model.concat_dropout(torch.cat([word_repr, char_repr, lang_repr], dim=-1))
            lstm_hidden = model.encoder(combined, mask=mask, return_hidden=True)
            attended_states, _ = model.attention(lstm_hidden, mask=mask)
            emissions = model.fc_shared(attended_states)

            loss = loss_fn(model.crf_shared, emissions, tags, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        dev_eval = evaluate_model_metrics(model, dev_path, "assamese", word_to_id, char_to_id, device)
        stress_eval = evaluate_model_metrics(model, stress_path, "assamese", word_to_id, char_to_id, device)
        s200_res = evaluate_sentence_200_emissions(model, s200_toks, s200_gold, word_to_id, char_to_id, device)

        print(f"  Ep {ep}: Dev F1={dev_eval['entity_f1']:.2f}% (P={dev_eval['entity_precision']:.2f}%, R={dev_eval['entity_recall']:.2f}%) | "
              f"Stress F1={stress_eval['entity_f1']:.2f}% (Collapse={stress_eval['collapse_rate']:.1f}%) | "
              f"S200 MeanGap={s200_res['mean_gap']:+.2f}, Tok0Gap={s200_res['token0_gap']:+.2f} | "
              f"S200 Ents={s200_res['pred_entity_tokens']}/8 (Hits={s200_res['hits']}) Spans={s200_res['pred_spans']}")

if __name__ == "__main__":
    train_c2 = gold_dir / "train_cycle2_corrected.conll"
    train_c2_v2 = gold_dir / "train_cycle2_corrected_v2.conll"
    test_epoch_trace("weight_2x", train_c2, 2.0)
    test_epoch_trace("weight_8x", train_c2, 8.0)
    test_epoch_trace("reinforced_data_1x", train_c2_v2, 1.0)
