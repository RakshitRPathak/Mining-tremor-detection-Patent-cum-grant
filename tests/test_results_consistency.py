"""The committed result files must be internally consistent, and the README must quote them correctly."""
import re

import pytest


def _pct(x):
    return f"{100 * x:.1f}%"


def test_confusion_counts_are_consistent_with_the_reported_rates(results):
    r = results["tremor_simulation_results"]
    for key in ("ml", "thr", "thr_smv_only", "hyb"):
        m = r[key]
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == r["n_test"]
        assert m["recall"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))
        assert m["fpr"] == pytest.approx(m["fp"] / (m["fp"] + m["tn"]))
        assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / (m["tn"] + m["fp"] + m["fn"] + m["tp"]))


def test_hybrid_can_only_remove_alerts_the_model_would_have_raised(results):
    r = results["tremor_simulation_results"]
    assert r["hyb"]["tp"] <= r["ml"]["tp"] and r["hyb"]["fp"] <= r["ml"]["fp"]      # RF AND threshold is a subset of RF


def test_dynamic_envelope_threshold_beats_smv_only_threshold(results):
    r = results["tremor_simulation_results"]
    assert r["thr"]["recall"] > r["thr_smv_only"]["recall"] and r["thr"]["accuracy"] > r["thr_smv_only"]["accuracy"]


def test_readme_detector_table_matches_the_results_file(results, readme):
    r = results["tremor_simulation_results"]
    rows = {"Threshold on SMV only": "thr_smv_only", "Threshold on dynamic envelope": "thr",
            "Random Forest only": "ml", "Hybrid (RF AND threshold)": "hyb"}
    for label, key in rows.items():
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([\d.]+%)\s*\|\s*([\d.]+%)\s*\|\s*([\d.]+%)\s*\|", readme)
        assert m, f"README row missing: {label}"
        assert m.groups() == (_pct(r[key]["accuracy"]), _pct(r[key]["recall"]), _pct(r[key]["fpr"])), label


def test_readme_headline_numbers_match_the_other_result_files(results, readme):
    ext = results["extended_analysis_results"]
    assert f"{100 * ext['mc']['acc']:.1f}% accuracy" in readme and f"macro F1 {ext['mc']['macro_f1']:.3f}" in readme
    assert f"{ext['lat']['median']:.2f} s median" in readme
    multi = results["tremor_simulation_results"]["multinode"]
    assert f"{100 * multi['shared_confirm']:.1f}% of shared events confirmed" in readme
    fw = results["streaming_and_firmware_results"]["doc7_8"]
    assert _pct(fw["current"]["delivery_rate"]) in readme
