import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.special import rel_entr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from dcor import distance_correlation

class MetricsCalculator:
    def _kl_divergence(self, p, q, epsilon=1e-10):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        p /= p.sum()
        q /= q.sum()
        return np.sum(rel_entr(p, q))

    def _jensen_shannon(self, p, q):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        m = (p + q) / 2
        return np.sqrt((self._kl_divergence(p, m) + self._kl_divergence(q, m)) / 2)

    def _tvd(self, p, q):
        return np.sum(np.abs(np.array(p) - np.array(q))) / 2

    def calculate_distribution_metrics(self, model_df, gold_df):
        metrics = {}
        gold_dist = np.vstack(gold_df['distribution'].to_list())
        model_dist = np.vstack(model_df['distribution'].to_list())
        
        metrics['KL_Divergence'] = np.mean([self._kl_divergence(p, q) for p, q in zip(model_dist, gold_dist)])
        metrics['Jensen_Shannon'] = np.mean([self._jensen_shannon(p, q) for p, q in zip(model_dist, gold_dist)])
        metrics['Total_Variation_Distance'] = np.mean([self._tvd(p, q) for p, q in zip(model_dist, gold_dist)])
        metrics['Distance_Correlation'] = distance_correlation(gold_dist, model_dist)
        return metrics

    def calculate_score_metrics(self, model_df, gold_df):
        metrics = {}
        gold_flat = gold_df['score'].explode().astype(float).to_list()
        model_flat = model_df['score'].explode().astype(float).to_list()
        
        metrics['RMSE_Avg'] = np.mean([root_mean_squared_error(g, p) for g, p in zip(gold_df['score'], model_df['score'])])
        metrics['MAE_Avg'] = np.mean([mean_absolute_error(g, p) for g, p in zip(gold_df['score'], model_df['score'])])
        metrics['R2_Score_Overall'] = r2_score(gold_flat, model_flat)
        return metrics

    def calculate_rank_metrics(self, model_df, gold_df):
        metrics = {}
        spearman_corrs, kendall_taus = [], []

        for gold_rank, model_rank in zip(gold_df['rank'], model_df['rank']):
            if len(set(gold_rank)) > 1 and len(set(model_rank)) > 1:
                spearman_corrs.append(spearmanr(gold_rank, model_rank)[0])
                kendall_taus.append(kendalltau(gold_rank, model_rank)[0])
        
        metrics['Spearman_Avg'] = np.nanmean(spearman_corrs)
        metrics['Kendall_Tau_Avg'] = np.nanmean(kendall_taus)
        return metrics

    def calculate_all_metrics(self, model_df, gold_df):
        all_metrics = {}
        if 'distribution' in model_df.columns and 'distribution' in gold_df.columns:
            all_metrics.update(self.calculate_distribution_metrics(model_df, gold_df))
        if 'score' in model_df.columns and 'score' in gold_df.columns:
            all_metrics.update(self.calculate_score_metrics(model_df, gold_df))
        if 'rank' in model_df.columns and 'rank' in gold_df.columns:
            # Gold standard needs a 'rank' column, which can be derived from 'score' or 'distribution'
            if 'rank' not in gold_df.columns and 'score' in gold_df.columns:
                gold_df['rank'] = gold_df['score'].apply(lambda x: np.argsort(-np.array(x)).tolist())
            if 'rank' in gold_df.columns:
                all_metrics.update(self.calculate_rank_metrics(model_df, gold_df))
        return all_metrics