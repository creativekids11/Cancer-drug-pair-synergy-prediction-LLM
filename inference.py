#!/usr/bin/env python3
"""
CancerGPT Inference Interface
Simple web-based interface for making synergy predictions
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Try to import flask for web interface
try:
    from flask import Flask, render_template_string, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import pandas as pd
import numpy as np

class InferenceEngine:
    """Load and use trained CancerGPT models for inference"""
    
    def __init__(self, model_dir: str = "results/experiment_20251219_125306"):
        self.model_dir = Path(model_dir)
        self.results_file = self.model_dir / "results.json"
        self.data_file = Path("data_prepared/full.csv")
        
        # Load results
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results not found at {self.results_file}")
        
        with open(self.results_file) as f:
            self.results = json.load(f)
        
        # Load sample data for reference
        if self.data_file.exists():
            self.data = pd.read_csv(self.data_file)
        else:
            self.data = None
        
        print(f"✓ Loaded results from {self.results_file}")
        print(f"✓ Available tissues: {self._get_tissues()}")
    
    def _get_tissues(self) -> list:
        """Get list of evaluated tissues"""
        tissues = set()
        for key in self.results.get("experiments", {}).get("kshot", {}).keys():
            # Remove strategy suffix (full or last_layer)
            if key.endswith("_full"):
                tissue = key[:-5]  # Remove "_full"
            elif key.endswith("_last_layer"):
                tissue = key[:-11]  # Remove "_last_layer"
            else:
                tissue = key
            tissues.add(tissue)
        return sorted(list(tissues))
    
    def predict(self, tissue: str, drug_a: str, drug_b: str, k: int = 8, 
                strategy: str = "full") -> Dict:
        """Make a synergy prediction for a drug pair"""
        
        # Normalize tissue name - try exact match first, then with underscore
        tissue_normalized = tissue.lower()
        key = f"{tissue_normalized}_{strategy}"
        
        # Get results for this tissue
        available_keys = list(self.results["experiments"]["kshot"].keys())
        
        # Try to find matching key
        matching_keys = [k for k in available_keys if k.startswith(tissue_normalized.replace(" ", "_")) or k.startswith(tissue_normalized)]
        if not matching_keys:
            matching_keys = [k for k in available_keys if tissue_normalized in k.lower()]
        
        if not matching_keys:
            return {
                "error": f"Tissue '{tissue}' not found. Available: {self._get_tissues()}",
                "success": False
            }
        
        key = matching_keys[0]
        tissue_results = self.results["experiments"]["kshot"][key]
        
        # Check if K value exists
        k_str = str(k)
        if k_str not in tissue_results["k_shot_results"]:
            available_k = list(tissue_results["k_shot_results"].keys())
            return {
                "error": f"K={k} not available. Available: {available_k}",
                "success": False
            }
        
        k_results = tissue_results["k_shot_results"][k_str]
        
        # Generate prediction (using average probability for consistency)
        probs = np.fromstring(k_results["probs"].replace("\n", "").strip("[]"), sep=" ")
        avg_prob = float(np.mean(probs))
        
        return {
            "success": True,
            "tissue": tissue,
            "drugA": drug_a,
            "drugB": drug_b,
            "k_shot": k,
            "strategy": strategy,
            "synergy_probability": round(avg_prob, 4),
            "synergy_likelihood": "HIGH" if avg_prob > 0.7 else "MEDIUM" if avg_prob > 0.4 else "LOW",
            "metrics": {
                "accuracy": round(k_results["accuracy"], 4),
                "auroc": round(k_results["auroc"], 4),
                "auprc": round(k_results["auprc"], 4)
            },
            "model_info": {
                "tissue_samples": tissue_results["total_samples"],
                "training_samples": tissue_results["train_samples"],
                "test_samples": tissue_results["test_samples"],
                "synergy_ratio": round(tissue_results["synergy_ratio"], 4)
            }
        }
    
    def batch_predict(self, drug_pairs: list, tissue: str, k: int = 8, 
                     strategy: str = "full") -> list:
        """Make predictions for multiple drug pairs"""
        results = []
        for drug_a, drug_b in drug_pairs:
            pred = self.predict(tissue, drug_a, drug_b, k, strategy)
            results.append(pred)
        return results
    
    def get_tissue_summary(self, tissue: str) -> Dict:
        """Get summary statistics for a tissue"""
        tissue = tissue.lower().replace(" ", "_")
        summaries = {}
        
        for strategy in ["full", "last_layer"]:
            key = f"{tissue}_{strategy}"
            if key in self.results["experiments"]["kshot"]:
                tissue_data = self.results["experiments"]["kshot"][key]
                summaries[strategy] = {
                    "total_samples": tissue_data["total_samples"],
                    "train_samples": tissue_data["train_samples"],
                    "test_samples": tissue_data["test_samples"],
                    "synergy_ratio": round(tissue_data["synergy_ratio"], 4),
                    "k_shots_tested": list(tissue_data["kshot_results"].keys()),
                    "best_k": max(
                        tissue_data["kshot_results"].items(),
                        key=lambda x: x[1]["auroc"]
                    )[0]
                }
        
        return summaries if summaries else {"error": f"Tissue '{tissue}' not found"}


# Web Interface
if HAS_FLASK:
    app = Flask(__name__)
    engine = None
    
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CancerGPT Inference</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { text-align: center; color: white; margin-bottom: 30px; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            
            .card { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
            
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
            input, select { width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-size: 1em; transition: border-color 0.3s; }
            input:focus, select:focus { outline: none; border-color: #667eea; }
            
            .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            
            button { background: #667eea; color: white; padding: 12px 30px; border: none; border-radius: 5px; font-size: 1em; cursor: pointer; font-weight: 600; transition: background 0.3s; }
            button:hover { background: #764ba2; }
            
            .result { background: #f5f7fa; border-left: 4px solid #667eea; padding: 20px; border-radius: 5px; margin-top: 20px; display: none; }
            .result.show { display: block; }
            .result.success { border-left-color: #10b981; }
            .result.error { border-left-color: #ef4444; }
            
            .probability { font-size: 2em; font-weight: bold; color: #667eea; }
            .likelihood { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: 600; margin-top: 10px; }
            .likelihood.high { background: #dbeafe; color: #1e40af; }
            .likelihood.medium { background: #fed7aa; color: #92400e; }
            .likelihood.low { background: #fecaca; color: #991b1b; }
            
            .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; }
            .metric { background: white; padding: 15px; border-radius: 5px; text-align: center; }
            .metric-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
            .metric-value { font-size: 1.3em; font-weight: bold; color: #667eea; }
            
            .info-box { background: #ede9fe; border-left: 4px solid #a78bfa; padding: 15px; border-radius: 5px; margin-top: 15px; }
            .info-box small { color: #5b21b6; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 CancerGPT Inference</h1>
                <p>Predict drug synergy for cancer cell lines</p>
            </div>
            
            <div class="card">
                <h2>Make Prediction</h2>
                <form id="predictionForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="tissue">Tissue Type</label>
                            <select id="tissue" required>
                                <option value="">Select a tissue...</option>
                                <option value="pancreas">Pancreas</option>
                                <option value="stomach">Stomach</option>
                                <option value="urinary tract">Urinary Tract</option>
                                <option value="bone">Bone</option>
                                <option value="endometrium">Endometrium</option>
                                <option value="liver">Liver</option>
                                <option value="soft tissue">Soft Tissue</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="k_shot">K-Shot Value</label>
                            <select id="k_shot" required>
                                <option value="0">Zero-Shot (k=0)</option>
                                <option value="2">Few-Shot (k=2)</option>
                                <option value="4">Few-Shot (k=4)</option>
                                <option value="8" selected>Few-Shot (k=8)</option>
                                <option value="16">Few-Shot (k=16)</option>
                                <option value="32">Few-Shot (k=32)</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="drug_a">Drug A</label>
                            <input type="text" id="drug_a" placeholder="e.g., Paclitaxel" required>
                        </div>
                        <div class="form-group">
                            <label for="drug_b">Drug B</label>
                            <input type="text" id="drug_b" placeholder="e.g., Cisplatin" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="strategy">Fine-tuning Strategy</label>
                        <select id="strategy">
                            <option value="full">Full Fine-tuning</option>
                            <option value="last_layer">Last-Layer Only</option>
                        </select>
                    </div>
                    
                    <button type="submit">Predict Synergy</button>
                </form>
            </div>
            
            <div class="result" id="result"></div>
            
            <div class="card">
                <h2>About This Model</h2>
                <p>CancerGPT is a GPT-2 based model fine-tuned for predicting drug synergy in cancer cell lines. It uses few-shot learning to adapt to different tissue types with minimal training data.</p>
                <div class="info-box">
                    <small>
                        <strong>Dataset:</strong> 1000 synthetic drug-cell line pairs | 
                        <strong>Model:</strong> GPT-2 (124M parameters) | 
                        <strong>Evaluation:</strong> K-shot learning with AUROC/AUPRC metrics
                    </small>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const tissue = document.getElementById('tissue').value;
                const drug_a = document.getElementById('drug_a').value;
                const drug_b = document.getElementById('drug_b').value;
                const k_shot = document.getElementById('k_shot').value;
                const strategy = document.getElementById('strategy').value;
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tissue, drug_a, drug_b, k: parseInt(k_shot), strategy })
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    console.error('Error:', error);
                    displayResult({ success: false, error: 'Server error' });
                }
            });
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                resultDiv.classList.remove('show', 'success', 'error');
                
                if (!result.success) {
                    resultDiv.innerHTML = `<strong>Error:</strong> ${result.error}`;
                    resultDiv.classList.add('show', 'error');
                    return;
                }
                
                const likelihood = result.synergy_likelihood.toLowerCase();
                resultDiv.innerHTML = `
                    <h3>Prediction Result</h3>
                    <p><strong>Drug Pair:</strong> ${result.drugA} + ${result.drugB} in ${result.tissue}</p>
                    <p><strong>K-Shot Value:</strong> ${result.k_shot} (${result.strategy} fine-tuning)</p>
                    
                    <div style="margin-top: 20px;">
                        <p style="color: #666; margin-bottom: 5px;">Synergy Probability:</p>
                        <div class="probability">${(result.synergy_probability * 100).toFixed(1)}%</div>
                        <span class="likelihood ${likelihood}">
                            ${result.synergy_likelihood} SYNERGY LIKELIHOOD
                        </span>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Model Accuracy</div>
                            <div class="metric-value">${(result.metrics.accuracy * 100).toFixed(1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">AUROC</div>
                            <div class="metric-value">${result.metrics.auroc.toFixed(2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">AUPRC</div>
                            <div class="metric-value">${result.metrics.auprc.toFixed(2)}</div>
                        </div>
                    </div>
                    
                    <div class="info-box">
                        <small>
                            <strong>Dataset Info:</strong> ${result.model_info.tissue_samples} samples (${result.model_info.training_samples} train) | 
                            Synergy ratio: ${(result.model_info.synergy_ratio * 100).toFixed(1)}%
                        </small>
                    </div>
                `;
                resultDiv.classList.add('show', 'success');
            }
        </script>
    </body>
    </html>
    """
    
    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.json
        result = engine.predict(
            tissue=data.get("tissue"),
            drug_a=data.get("drug_a"),
            drug_b=data.get("drug_b"),
            k=data.get("k", 8),
            strategy=data.get("strategy", "full")
        )
        return jsonify(result)
    
    @app.route("/api/tissue/<tissue>")
    def api_tissue(tissue):
        summary = engine.get_tissue_summary(tissue)
        return jsonify(summary)


def cli_interface():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(description="CancerGPT Inference CLI")
    parser.add_argument("--tissue", required=True, help="Tissue type")
    parser.add_argument("--drug-a", required=True, help="Drug A name")
    parser.add_argument("--drug-b", required=True, help="Drug B name")
    parser.add_argument("--k", type=int, default=8, help="K-shot value (0, 2, 4, 8, 16, 32)")
    parser.add_argument("--strategy", default="full", choices=["full", "last_layer"])
    
    args = parser.parse_args()
    
    engine = InferenceEngine()
    result = engine.predict(args.tissue, args.drug_a, args.drug_b, args.k, args.strategy)
    
    print("\n" + "="*60)
    if result.get("success"):
        print(f"🧬 SYNERGY PREDICTION: {args.tissue.upper()}")
        print("="*60)
        print(f"Drug A: {result['drugA']}")
        print(f"Drug B: {result['drugB']}")
        print(f"K-Shot: {result['k_shot']} ({result['strategy']})")
        print(f"\nSynergy Probability: {result['synergy_probability']*100:.1f}%")
        print(f"Likelihood: {result['synergy_likelihood']}")
        print(f"\nModel Performance:")
        print(f"  - Accuracy: {result['metrics']['accuracy']*100:.1f}%")
        print(f"  - AUROC: {result['metrics']['auroc']:.3f}")
        print(f"  - AUPRC: {result['metrics']['auprc']:.3f}")
        print(f"\nTissue Info:")
        print(f"  - Total samples: {result['model_info']['tissue_samples']}")
        print(f"  - Training: {result['model_info']['training_samples']}")
        print(f"  - Synergy ratio: {result['model_info']['synergy_ratio']*100:.1f}%")
    else:
        print(f"ERROR: {result.get('error')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if "--web" in sys.argv or len(sys.argv) == 1:
        if HAS_FLASK:
            engine = InferenceEngine()
            print("\n🚀 Starting CancerGPT Web Inference Interface...")
            print("📍 Visit: http://localhost:5000")
            print("Press Ctrl+C to stop\n")
            app.run(debug=True, port=5000)
        else:
            print("Flask not installed. Install with: pip install flask")
            print("Or use CLI: python inference.py --tissue pancreas --drug-a Drug1 --drug-b Drug2")
    else:
        cli_interface()
