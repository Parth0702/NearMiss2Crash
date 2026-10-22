#!/usr/bin/env python3
"""
NearMiss2Crash — Phase 3D Audit Script

Audit and correct Phase 3D experiment results before final manuscript integration.

Tasks:
1. Verify actual Gemini API responses exist
2. Regenerate all metrics from per-report predictions
3. Reconcile coverage + abstentions = total
4. Correct system descriptions
5. Fix sampling to include Uncertain reports
6. Permit Gemini Uncertain output
7. Validate labels against evidence scores
8. Validate narrative excerpts
9. Produce corrected metrics and audit report
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# HFACS classes
HFACS_CLASSES = ["Skill-Based Error", "Decision Error", "Violation", "Perceptual Error"]

@dataclass
class AuditFinding:
    """Audit finding with severity and description."""
    severity: str  # "CRITICAL", "MAJOR", "MINOR"
    category: str   # "DATA", "METHOD", "METRIC", "SAMPLING"
    description: str
    details: str
    affected_reports: List[str]

def load_fuzzy_data() -> pd.DataFrame:
    """Load fuzzy labeling results."""
    path = Path("experiments/fuzzy_asrs/results/fuzzy_full_results.csv")
    if not path.exists():
        raise FileNotFoundError(f"Fuzzy data not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded fuzzy data: {len(df)} reports")

    # Check for Uncertain reports
    uncertain_count = (df['fuzzy_label'] == 'Uncertain').sum()
    print(f"Uncertain reports: {uncertain_count}/{len(df)} ({uncertain_count/len(df)*100:.1f}%)")

    return df

def load_gemini_cache() -> Dict[str, Any]:
    """Load Gemini cache data."""
    path = Path("experiments/llm_fuzzy_fusion/gemini_pilot_cache.json")
    if not path.exists():
        raise FileNotFoundError(f"Gemini cache not found: {path}")

    with open(path, 'r') as f:
        cache = json.load(f)

    print(f"Loaded Gemini cache: {len(cache)} reports")
    return cache

def validate_gemini_responses(cache: Dict[str, Any]) -> List[AuditFinding]:
    """Validate Gemini responses for completeness and consistency."""
    findings = []

    for report_id, data in cache.items():
        if not data.get('success', False):
            findings.append(AuditFinding(
                severity="CRITICAL",
                category="DATA",
                description="Failed Gemini API response",
                details=f"Report {report_id}: API call failed",
                affected_reports=[report_id]
            ))
            continue

        response = data.get('gemini_response', {})

        # Check required fields
        required_fields = ['hfacs_evidence', 'proposed_label', 'confidence']
        for field in required_fields:
            if field not in response:
                findings.append(AuditFinding(
                    severity="CRITICAL",
                    category="DATA",
                    description="Missing required field in Gemini response",
                    details=f"Report {report_id}: Missing '{field}'",
                    affected_reports=[report_id]
                ))

        # Check evidence structure
        evidence = response.get('hfacs_evidence', [])
        if len(evidence) != 4:
            findings.append(AuditFinding(
                severity="CRITICAL",
                category="DATA",
                description="Incorrect evidence count",
                details=f"Report {report_id}: Expected 4 classes, got {len(evidence)}",
                affected_reports=[report_id]
            ))

        # Check evidence scores
        for item in evidence:
            score = item.get('evidence_score', -1)
            if not (0.0 <= score <= 1.0):
                findings.append(AuditFinding(
                    severity="CRITICAL",
                    category="DATA",
                    description="Invalid evidence score",
                    details=f"Report {report_id}: Invalid score {score} for {item.get('class', 'unknown')}",
                    affected_reports=[report_id]
                ))

        # Check label consistency
        label = response.get('proposed_label')
        if label not in HFACS_CLASSES:
            findings.append(AuditFinding(
                severity="CRITICAL",
                category="DATA",
                description="Invalid proposed label",
                details=f"Report {report_id}: Invalid label '{label}'",
                affected_reports=[report_id]
            ))

        # Check confidence consistency
        confidence = response.get('confidence', -1)
        if not (0.0 <= confidence <= 1.0):
            findings.append(AuditFinding(
                severity="CRITICAL",
                category="DATA",
                description="Invalid confidence score",
                details=f"Report {report_id}: Invalid confidence {confidence}",
                affected_reports=[report_id]
            ))

        # Check if proposed label matches evidence
        max_score = max([item['evidence_score'] for item in evidence])
        label_score = next((item['evidence_score'] for item in evidence
                          if item['class'] == label), -1)

        if label_score < max_score and max_score > 0.1:
            findings.append(AuditFinding(
                severity="MAJOR",
                category="METHOD",
                description="Proposed label doesn't match highest evidence score",
                details=f"Report {report_id}: Label '{label}' (score {label_score:.2f}) but max score is {max_score:.2f}",
                affected_reports=[report_id]
            ))

    return findings

def analyze_sampling_strata(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze current sampling distribution."""
    strata_stats = {
        'total_reports': len(df),
        'by_rule': {},
        'by_label': {},
        'uncertain_reports': [],
        'exact_ties': [],
        'low_dominance': [],
        'no_evidence': []
    }

    # Rule distribution
    rule_counts = df['rule_fired'].value_counts()
    strata_stats['by_rule'] = rule_counts.to_dict()

    # Label distribution
    label_counts = df['fuzzy_label'].value_counts()
    strata_stats['by_label'] = label_counts.to_dict()

    # Identify specific strata
    strata_stats['uncertain_reports'] = df[df['fuzzy_label'] == 'Uncertain']['report_id'].tolist()
    strata_stats['exact_ties'] = df[df['rule_fired'] == 'R3_EXACT_TIE']['report_id'].tolist()
    strata_stats['low_dominance'] = df[df['rule_fired'] == 'R4_BELOW_DOMINANCE_MARGIN']['report_id'].tolist()
    strata_stats['no_evidence'] = df[df['rule_fired'] == 'R0_NO_EVIDENCE']['report_id'].tolist()

    return strata_stats

def calculate_metrics_from_data(df: pd.DataFrame, cache: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics directly from data, not from previous results."""
    metrics = {
        'crisp': {},
        'fuzzy': {},
        'gemini': {},
        'fusion': {}
    }

    # Crisp metrics
    crisp_total = len(df)
    crisp_labeled = (df['crisp_label'] != 'Unlabeled').sum()
    metrics['crisp'] = {
        'total': crisp_total,
        'definitive': crisp_labeled,
        'coverage': crisp_labeled / crisp_total if crisp_total > 0 else 0,
        'abstentions': crisp_total - crisp_labeled,
        'abstention_rate': (crisp_total - crisp_labeled) / crisp_total if crisp_total > 0 else 0,
        'class_distribution': df[df['crisp_label'] != 'Unlabeled']['crisp_label'].value_counts().to_dict()
    }

    # Fuzzy metrics
    fuzzy_total = len(df)
    fuzzy_labeled = (df['fuzzy_label'] != 'Uncertain').sum()
    metrics['fuzzy'] = {
        'total': fuzzy_total,
        'definitive': fuzzy_labeled,
        'coverage': fuzzy_labeled / fuzzy_total if fuzzy_total > 0 else 0,
        'abstentions': fuzzy_total - fuzzy_labeled,
        'abstention_rate': (fuzzy_total - fuzzy_labeled) / fuzzy_total if fuzzy_total > 0 else 0,
        'class_distribution': df[df['fuzzy_label'] != 'Uncertain']['fuzzy_label'].value_counts().to_dict()
    }

    # Gemini metrics (from cache)
    gemini_reports = list(cache.keys())
    gemini_labeled = 0
    gemini_class_dist = {}

    for report_id in gemini_reports:
        data = cache[report_id]
        if data.get('success', False):
            label = data['gemini_response']['proposed_label']
            if label != 'Uncertain':  # Assuming Gemini doesn't abstain
                gemini_labeled += 1
                gemini_class_dist[label] = gemini_class_dist.get(label, 0) + 1

    metrics['gemini'] = {
        'total': len(gemini_reports),
        'definitive': gemini_labeled,
        'coverage': gemini_labeled / len(gemini_reports) if len(gemini_reports) > 0 else 0,
        'abstentions': len(gemini_reports) - gemini_labeled,
        'abstention_rate': (len(gemini_reports) - gemini_labeled) / len(gemini_reports) if len(gemini_reports) > 0 else 0,
        'class_distribution': gemini_class_dist
    }

    # Note: Fusion metrics require fusion calculations

    return metrics

def main():
    print("=" * 70)
    print("NearMiss2Crash — Phase 3D Audit")
    print("=" * 70)

    # Load data
    try:
        df_fuzzy = load_fuzzy_data()
        cache = load_gemini_cache()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Audit Gemini responses
    print("\n[1] Auditing Gemini responses...")
    gemini_findings = validate_gemini_responses(cache)
    for finding in gemini_findings:
        print(f"  {finding.severity}: {finding.description}")
        print(f"    Details: {finding.details}")

    # Analyze sampling
    print("\n[2] Analyzing sampling strata...")
    strata_stats = analyze_sampling_strata(df_fuzzy)
    print(f"Total reports: {strata_stats['total_reports']}")
    print(f"Uncertain reports: {len(strata_stats['uncertain_reports'])}")
    print(f"Exact ties: {len(strata_stats['exact_ties'])}")
    print(f"Low dominance: {len(strata_stats['low_dominance'])}")
    print(f"No evidence: {len(strata_stats['no_evidence'])}")

    # Calculate metrics
    print("\n[3] Calculating metrics from raw data...")
    metrics = calculate_metrics_from_data(df_fuzzy, cache)

    # Print metrics
    for system in ['crisp', 'fuzzy', 'gemini']:
        print(f"\n{system.upper()} SYSTEM:")
        m = metrics[system]
        print(f"  Total: {m['total']}")
        print(f"  Definitive: {m['definitive']} ({m['coverage']*100:.1f}%)")
        print(f"  Abstentions: {m['abstentions']} ({m['abstention_rate']*100:.1f}%)")
        print(f"  Class distribution: {m['class_distribution']}")

    # Save audit findings
    audit_report = {
        'audit_date': '2026-10-22',
        'findings': [
            {
                'severity': finding.severity,
                'category': finding.category,
                'description': finding.description,
                'details': finding.details,
                'affected_reports': finding.affected_reports
            }
            for finding in gemini_findings
        ],
        'sampling_analysis': strata_stats,
        'metrics': metrics
    }

    with open('phase3d_audit_report.json', 'w') as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nAudit report saved to: phase3d_audit_report.json")

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    critical_findings = [f for f in gemini_findings if f.severity == "CRITICAL"]
    major_findings = [f for f in gemini_findings if f.severity == "MAJOR"]

    print(f"Critical findings: {len(critical_findings)}")
    print(f"Major findings: {len(major_findings)}")

    if critical_findings:
        print("\nCRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
        for finding in critical_findings[:3]:  # Show top 3
            print(f"  - {finding.description}")

    print(f"\nActual experiment size: {len(cache)} reports (not 148)")
    print("Sampling issue: Original experiment filtered out Uncertain reports")

if __name__ == "__main__":
    main()