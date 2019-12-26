import argparse
import json
from collections import defaultdict
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print('{}Analyze train time of {}{}'.format('-' * 5, args.json_logs[i],
                                                    '-' * 5))
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print('slowest epoch {}, average time is {:.4f}'.format(
            slowest_epoch + 1, epoch_ave_time[slowest_epoch]))
        print('fastest epoch {}, average time is {:.4f}'.format(
            fastest_epoch + 1, epoch_ave_time[fastest_epoch]))
        print('time std over epochs is {:.4f}'.format(std_over_epoch))
        print('average iter time: {:.4f} s/iter'.format(np.mean(all_times)))
        print()


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                # legend.append('{}_{}'.format(json_log, metric))
                legend.append('{}'.format( metric))
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    #这里只要 loss_bbox ,loss_cls, lr,acc四个metric
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())

        host=host_subplot(111,axes_class=AA.Axes)#host这里可以表示 bbox loss和clsloss
        plt.subplots_adjust(right=0.75)
        acc1=host.twinx()
        lr1=host.twinx()
        offset=0
        new_fixed_axis=lr1.get_grid_helper().new_fixed_axis
        lr1.axis['right']=new_fixed_axis(loc='right',axes=lr1,offset=(offset,0))
        lr1.axis['right'].toggle(all=True)
        host.set_ylim(0,2)
        host.set_xlabel("iter")
        host.set_ylabel("loss")
        lr1.set_ylabel("lr")
        # lr1.set_ylabel("lr")
        xs_lr = []
        ys_lr = []
        xs_bbox_loss=[]
        ys_bbox_loss=[]
        # xs_cls_loss=[]
        ys_cls_loss=[]

        num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
        for epoch in epochs:
            iters = log_dict[epoch]['iter']
            if log_dict[epoch]['mode'][-1] == 'val':
                iters = iters[:-1]
            xs_lr.append(
                np.array(iters) + (epoch - 1) * num_iters_per_epoch)
            ys_lr.append(np.array(log_dict[epoch]['lr'][:len(iters)]))

            xs_bbox_loss.append(
                np.array(iters) + (epoch - 1) * num_iters_per_epoch)
            ys_bbox_loss.append(np.array(log_dict[epoch]['loss_bbox'][:len(iters)]))
            # xs_cls_loss.append(
            #     np.array(iters) + (epoch - 1) * num_iters_per_epoch)
            ys_cls_loss.append(np.array(log_dict[epoch]['loss_cls'][:len(iters)]))
        xs_lr = np.concatenate(xs_lr)
        ys_lr = np.concatenate(ys_lr)
        ys_cls_loss=np.concatenate(ys_cls_loss)
        ys_bbox_loss=np.concatenate(ys_bbox_loss)
        p1,=host.plot(xs_lr,ys_cls_loss,label='loss_cls')
        p2,=host.plot(xs_lr,ys_bbox_loss,label='loss_bbox')
        p_lr, = lr1.plot(
            xs_lr, ys_lr, label='lr', linewidth=0.5)
        lr1.set_ylim(0,0.01)
        host.legend()
        host.axis['left'].label.set_color(p1.get_color())
        host.axis['left'].label.set_color(p2.get_color())
        lr1.axis['right'].label.set_color(p_lr.get_color())
        plt.draw()
        plt.show()




    if args.title is not None:
        plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print('save curve to: {}'.format(args.out))
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for l in log_file:
                log = json.loads(l.strip())
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
