# coding=utf8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from pts.model.tempflow import TempFlowEstimator
from pts.model.time_grad import TimeGradEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    label_prefix = ""
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape

    ps = [50.0] + [
            50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
        ]

    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length :][dim].plot(ax=ax)

        ps_data = [forecast.quantile(p / 100.0)[:,dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-",
                        label=f"{label_prefix}median", ax=ax)

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]    
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)


def train():
    import ipdb; ipdb.set_trace()
    dataset = get_dataset("electricity_nips", regenerate=False)
    train_grouper = MultivariateGrouper(max_target_dim=min(2000,
        int(dataset.metadata.feat_static_cat[0].cardinality)))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)),
                                       max_target_dim=min(2000,
                                                          int(dataset.metadata.feat_static_cat[0].cardinality)))
    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)
    estimator = TimeGradEstimator(
                target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
                prediction_length=dataset.metadata.prediction_length,
                context_length=dataset.metadata.prediction_length,
                cell_type='GRU',
                input_size=1484,
                freq=dataset.metadata.freq,
                loss_type='l2',
                scaling=True,
                diff_steps=100,
                beta_end=0.1,
                beta_schedule="linear",
                trainer=Trainer(device=device,
                                epochs=5,
                                learning_rate=1e-3,
                                num_batches_per_epoch=100,
                                batch_size=64,)
    )
    predictor = estimator.train(dataset_train, num_workers=0)



if __name__ == '__main__':
    train()
