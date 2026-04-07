# Link Check Report

**Generated:** 2026-04-06  
**Scope:** `docs/training/*.md`, `notebooks/*.ipynb`, `examples/training/*.yaml`, `examples/training/README.md`  
**Method:** HTTP HEAD requests via `curl -s -o /dev/null -w "%{http_code}" --max-time 15 -L`

---

## Summary

| Status | Count |
|--------|-------|
| ✅ 200 OK | 18 |
| ❌ 404 Not Found | 3 |
| ⚠️ 401 Unauthorized | 1 |
| ⚠️ Needs Verification | 1 |

---

## ✅ Working Links (200 OK)

| URL | Source File | Notes |
|-----|-------------|-------|
| `https://github.com/dscripka/openWakeWord` | `docs/training/troubleshooting.md`, `docs/training/integration-guide.md`, `docs/training/data-format-reference.md`, `notebooks/choosing-your-training-harness.ipynb` | Primary openWakeWord repo |
| `https://github.com/OHF-Voice/micro-wake-word` | `docs/training/integration-guide.md`, `notebooks/choosing-your-training-harness.ipynb` | Primary micro-wake-word repo |
| `https://github.com/esphome/micro-wake-word-models` | `docs/training/integration-guide.md`, `notebooks/choosing-your-training-harness.ipynb` | Pre-trained models |
| `https://esphome.io/components/micro_wake_word` | `docs/training/integration-guide.md`, `notebooks/training-with-microwakeword.ipynb`, `notebooks/choosing-your-training-harness.ipynb` | ESPHome integration docs |
| `https://esphome.io/components/micro_wake_word.html` | `docs/training/integration-guide.md` | Same page as above (200, both work) |
| `https://github.com/dscripka/openWakeWord#readme` | `docs/training/integration-guide.md`, `notebooks/choosing-your-training-harness.ipynb` | openWakeWord readme |
| `https://github.com/kahrendt/microWakeWord` | `notebooks/training-with-microwakeword.ipynb` | **Correct** microWakeWord repo |
| `https://github.com/puddly/pymicro-features` | `notebooks/training-with-microwakeword.ipynb` | Git pip install URL |
| `https://github.com/whatsnowplaying/audio-metadata` | `notebooks/training-with-microwakeword.ipynb` | Git pip install URL |
| `https://download.pytorch.org/whl/cpu` | `notebooks/training-with-openwakeword.ipynb` | PyTorch CPU index |
| `https://github.com/OHF-Voice/micro-wake-word/blob/main/notebooks/basic_training_notebook.ipynb` | Key URLs check | |
| `https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb` | Key URLs check | |
| `https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb` | Key URLs check | |
| `https://huggingface.co/datasets/davidscripka/openwakeword_features` | Key URLs check | |
| `https://github.com/esphome/micro-wake-word-models` (clone URL) | `docs/training/integration-guide.md` | |
| `https://github.com/OHF-Voice/micro-wake-word` (clone URL) | `docs/training/integration-guide.md` | |
| `https://github.com/dscripka/openWakeWord` (clone URL) | `docs/training/integration-guide.md` | |
| `https://huggingface.co/datasets/kahrendt/microwakeword` (page) | Key URLs check | Dataset page 200; specific file paths may vary |

---

## ❌ Broken Links (404 Not Found)

### 1. `https://github.com/kahara/microWakeWord`

| | |
|--|--|
| **Found in** | `docs/training/troubleshooting.md` (line 1490), `docs/training/data-format-reference.md` (line 607) |
| **Status** | `404` |
| **Fix required** | Change `kahara` → `kahrendt` |
| **Correct URL** | `https://github.com/kahrendt/microWakeWord` |
| **Severity** | **HIGH** — affects 2 documentation files |
| **Note** | The `kahrendt` username is the correct one (Kevin Ahrendt), author of microWakeWord. No redirect exists from `kahara`. |

### 2. `https://github.com/rosticci/microWakeWord`

| | |
|--|--|
| **Found in** | `examples/training/README.md` (line 206) |
| **Status** | `404` |
| **Fix required** | Change `rosticci` → `kahrendt` |
| **Correct URL** | `https://github.com/kahrendt/microWakeWord` |
| **Severity** | **HIGH** — affects README.md "Further Reading" section |
| **Note** | `rosticci` is not a valid GitHub username for this project. |

### 3. `https://github.com/your-repo/wakeword-workbench`

| | |
|--|--|
| **Found in** | `notebooks/training-with-microwakeword.ipynb` (line 1284) |
| **Status** | `404` |
| **Fix required** | Replace with actual repository URL |
| **Severity** | **MEDIUM** — placeholder link in notebook documentation |
| **Note** | This is a template placeholder. Should be updated with the actual repo URL once published. |

---

## ⚠️ Needs Investigation / Special Cases

### 1. `https://huggingface.co/datasets/openwakeword/ACAV100M_sample`

| | |
|--|--|
| **Found in** | `examples/training/openwakeword-production.yaml` (line 40) |
| **Status** | `401 Unauthorized` |
| **Severity** | **HIGH** — this is a referenced negative dataset |
| **Recommended action** | Verify this dataset still exists. It may have been: (a) made private/deleted, (b) moved to a different location, or (c) requires authentication |
| **Alternative** | Check `https://huggingface.co/datasets` for openwakeword datasets, or consult the openWakeWord documentation |

### 2. `https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/`

| | |
|--|--|
| **Found in** | `notebooks/training-with-microwakeword.ipynb` (line 662) |
| **Status** | `401` on bare `/resolve/main/`; specific file 404 |
| **Dataset page** | `https://huggingface.co/datasets/kahrendt/microwakeword` — **200 OK** |
| **Severity** | **LOW** — the dataset exists and is public; the notebook may need a specific file path rather than the directory root |
| **Recommended action** | Verify the notebook uses the correct file paths for this dataset |

---

## Broken Link Fixes Required

### File: `docs/training/troubleshooting.md`
```diff
- https://github.com/kahara/microWakeWord
+ https://github.com/kahrendt/microWakeWord
```

### File: `docs/training/data-format-reference.md`
```diff
- https://github.com/kahara/microWakeWord
+ https://github.com/kahrendt/microWakeWord
```

### File: `examples/training/README.md`
```diff
- [microWakeWord Documentation](https://github.com/rosticci/microWakeWord)
+ [microWakeWord Documentation](https://github.com/kahrendt/microWakeWord)
```

### File: `notebooks/training-with-microwakeword.ipynb`
```diff
- [WakeWord Workbench Documentation](https://github.com/your-repo/wakeword-workbench)
+ [WakeWord Workbench Documentation](https://github.com/<actual-owner>/wakeword-workbench)
```
*Replace `<actual-owner>` with the real GitHub organization or username once the repo is public.*

### File: `examples/training/openwakeword-production.yaml`
- **Action required:** Verify `https://huggingface.co/datasets/openwakeword/ACAV100M_sample` is still available. If not, find an alternative negative dataset source and update the comment.

---

## Version Pins Documented

| URL | Pin Used | Notes |
|-----|----------|-------|
| `https://github.com/whatsnowplaying/audio-metadata` | `d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f` | Commit hash pin used in notebook |
| `https://github.com/puddly/pymicro-features` | `puddly/minimum-cpp-version` | Branch/reference pin used in notebook |
| PyTorch | `https://download.pytorch.org/whl/cpu` | No version pin on index URL; recommend pinning version in notebook cells |

---

## Repository References Summary

| Repository | Status | Correct Name |
|------------|--------|--------------|
| microWakeWord (Kevin Ahrendt) | ✅ | `kahrendt/microWakeWord` |
| microWakeWord (OHF-Voice fork) | ✅ | `OHF-Voice/micro-wake-word` |
| microWakeWord (esphome models) | ✅ | `esphome/micro-wake-word-models` |
| openWakeWord | ✅ | `dscripka/openWakeWord` |
| microWakeWord (kahara — WRONG) | ❌ | Should be `kahrendt` |
| microWakeWord (rosticci — WRONG) | ❌ | Should be `kahrendt` |

---

## Recommendations

1. **Fix broken links immediately** — 3 GitHub repo links and 1 placeholder need correction
2. **Verify HuggingFace dataset** — `openwakeword/ACVA100M_sample` needs investigation (401)
3. **Pin repo URL** — Add the actual GitHub repo URL for `wakeword-workbench` once known
4. **Add link checker to CI** — Consider adding a pre-commit or CI step that validates external links
