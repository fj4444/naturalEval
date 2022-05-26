## for cluster_gcn.py
import torch


class Logger(object):
    def __init__(self, runs, info=None, log=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.log = log

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print(self, mess):
        if self.log is not None:
            self.log.info(mess)
        else:
            print(mess)

    def print_statistics(self, run=None):
        if run is not None:
            if self.results[run] == []:
                self.print(f'WARNING: results from run {run} not recorded')
                return
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            self.print(f'Run {run + 1:02d}:')
            self.print(f'Highest Train: {result[:, 0].max():.2f}')
            self.print(f'Highest Valid: {result[:, 1].max():.2f}')
            self.print(f'  Final Train: {result[argmax, 0]:.2f}')
            self.print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            results_recorded = [line for line in self.results if line != []]
            if len(results_recorded) < len(self.results):
                self.print(f'WARNING: results from only {len(results_recorded)}/{len(self.results)} runs are recorded')
            result = 100 * torch.tensor(results_recorded)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            self.print(f'All runs:')
            r = best_result[:, 0]
            self.print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            self.print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            self.print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            self.print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
