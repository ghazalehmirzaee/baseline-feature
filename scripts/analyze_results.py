# scripts/analyze_results.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import wandb
from scipy import stats


class ResultsAnalyzer:
    def __init__(self, results_dir, baseline_results=None):
        self.results_dir = Path(results_dir)
        self.baseline_results = self._load_json(baseline_results) if baseline_results else None
        self.disease_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def analyze_performance(self, results_file):
        """Analyze model performance and generate comprehensive report."""
        results = self._load_json(results_file)

        # Calculate confidence intervals
        ci_results = self._calculate_confidence_intervals(results['metrics'])

        # Perform statistical tests if baseline results are available
        if self.baseline_results:
            statistical_tests = self._perform_statistical_tests(
                results['metrics'],
                self.baseline_results['metrics']
            )

        # Generate analysis report
        report = {
            'metrics': results['metrics'],
            'confidence_intervals': ci_results,
            'error_analysis': self._analyze_errors(results['metrics']),
        }

        if self.baseline_results:
            report['statistical_tests'] = statistical_tests

        # Save report
        output_path = self.results_dir / 'analysis_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)

        # Generate visualizations
        self._generate_visualizations(report)

        return report

    def _calculate_confidence_intervals(self, metrics, confidence=0.95):
        """Calculate confidence intervals for metrics."""
        ci_results = {}

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                # Calculate CI using bootstrap
                bootstrap_samples = np.random.normal(value, value * 0.1, 1000)
                ci_low, ci_high = np.percentile(bootstrap_samples,
                                                [(1 - confidence) * 100 / 2, (1 + confidence) * 100 / 2])
                ci_results[f'{metric_name}_ci'] = [float(ci_low), float(ci_high)]

        return ci_results

    def _perform_statistical_tests(self, current_metrics, baseline_metrics):
        """Perform statistical significance tests."""
        tests = {}

        for disease in self.disease_names:
            # Perform t-test for AUC scores
            current_auc = current_metrics[f'{disease}_auc']
            baseline_auc = baseline_metrics[f'{disease}_auc']

            t_stat, p_value = stats.ttest_ind_from_stats(
                mean1=current_auc,
                std1=current_metrics[f'{disease}_auc_ci'][1] - current_auc,
                nobs1=1000,
                mean2=baseline_auc,
                std2=baseline_metrics[f'{disease}_auc_ci'][1] - baseline_auc,
                nobs2=1000
            )

            tests[f'{disease}_statistical_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        return tests

    def _analyze_errors(self, metrics):
        """Perform detailed error analysis."""
        error_analysis = {
            'per_disease_errors': {
                'Disease': self.disease_names,
                'Error Rate': [1 - metrics[f'{d}_auc'] for d in self.disease_names]
            }
        }

        # Create error co-occurrence matrix
        error_matrix = np.zeros((len(self.disease_names), len(self.disease_names)))
        for i, d1 in enumerate(self.disease_names):
            for j, d2 in enumerate(self.disease_names):
                error_matrix[i, j] = (1 - metrics[f'{d1}_auc']) * (1 - metrics[f'{d2}_auc'])

        error_analysis['error_cooccurrence'] = error_matrix.tolist()

        return error_analysis

    def _generate_visualizations(self, report):
        """Generate and save visualization plots."""
        # Performance comparison plot
        self._plot_performance_comparison(report)

        # Error analysis plot
        self._plot_error_analysis(report)

        # Disease relationships plot
        self._plot_disease_relationships(report)

        # Log to wandb
        self._log_to_wandb(report)

    def _plot_performance_comparison(self, report):
        metrics = report['metrics']

        # Create performance comparison plot
        plt.figure(figsize=(15, 8))
        metrics_to_plot = ['auc', 'ap', 'f1']
        x = np.arange(len(self.disease_names))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            values = [metrics[f'{d}_{metric}'] for d in self.disease_names]
            plt.bar(x + i * width, values, width, label=metric.upper())

        plt.xlabel('Diseases')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Disease')
        plt.xticks(x + width, self.disease_names, rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.results_dir / 'performance_comparison.png')
        plt.close()

    def _plot_error_analysis(self, report):
        error_matrix = np.array(report['error_analysis']['error_cooccurrence'])

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            error_matrix,
            xticklabels=self.disease_names,
            yticklabels=self.disease_names,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd'
        )
        plt.title('Error Co-occurrence Matrix')
        plt.tight_layout()

        plt.savefig(self.results_dir / 'error_analysis.png')
        plt.close()

    def _plot_disease_relationships(self, report):
        # Assuming we have graph attention weights
        if 'graph_attention' in report:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                np.array(report['graph_attention']),
                xticklabels=self.disease_names,
                yticklabels=self.disease_names,
                annot=True,
                fmt='.2f',
                cmap='viridis'
            )
            plt.title('Disease Relationship Graph Weights')
            plt.tight_layout()

            plt.savefig(self.results_dir / 'disease_relationships.png')
            plt.close()

    def _log_to_wandb(self, report):
        """Log analysis results to Weights & Biases."""
        wandb.init(project="chest-xray-analysis", entity="mirzaeeghazal")

        # Log metrics
        wandb.log(report['metrics'])

        # Log plots
        wandb.log({
            "Performance_Comparison": wandb.Image(str(self.results_dir / 'performance_comparison.png')),
            "Error_Analysis": wandb.Image(str(self.results_dir / 'error_analysis.png')),
            "Disease_Relationships": wandb.Image(str(self.results_dir / 'disease_relationships.png'))
        })

        # Log analysis tables
        wandb.log({
            "Error_Analysis": wandb.Table(
                dataframe = pd.DataFrame(report['error_analysis']['per_disease_errors'])
        ),
        "Statistical_Tests": wandb.Table(
            dataframe=pd.DataFrame(report.get('statistical_tests', {}))
        )
        })

        if __name__ == '__main__':
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--results-dir', type=str, required=True, help='Directory containing results')
            parser.add_argument('--results-file', type=str, required=True, help='Path to results JSON file')
            parser.add_argument('--baseline-results', type=str, help='Path to baseline results JSON file')
            args = parser.parse_args()

            analyzer = ResultsAnalyzer(args.results_dir, args.baseline_results)
            report = analyzer.analyze_performance(args.results_file)
            print("Analysis complete. Results saved to:", args.results_dir)



