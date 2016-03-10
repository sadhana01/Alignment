import os
import json
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import warnings


from collections import defaultdict
from os.path import join
import entailment
import ilp_config

__author__ = 'chetannaik'

# Force matplotlib to not use any Xwindows backend (to run without error on ambiguity)
matplotlib.use('Agg')
ilp_config.set_plot_config()


def load_srl_data(srl_file):
    """Read the srl json, parse it into a python dictionary and return."""
    d = json.load(open(srl_file, "r"))
    data = {}
    for p_data in d:
        process = p_data['process']
        ss_data = p_data['sentences']
        sent_to_id = {}
        id_to_args = {}
        sentences={}
        arg_role_scores = {}
        arg_role_srl_data = {}
        for s_data in ss_data:
            sentence = s_data['text']
            s_id = s_data['sentenceId']
            sent_to_id[sentence] = s_id
            sentences[s_id]=sentence
            a_spans = s_data['predictionArgumentSpan']
            args = []
            if len(a_spans) != 0:
                for a_span in a_spans:
                    srl_role_prediction = a_span['rolePredicted']
                    start_idx = a_span['startIdx']
                    end_idx = a_span['endIdx']
                    arg_text = a_span['text']
                    arg_id = a_span['argId']
                    role_prob_list = a_span['probRoles']
                    args.append((arg_id, arg_text))
                    role_probs = {}
                    for role_prob in role_prob_list:
                        role_probs.update(role_prob)
                    arg_role_scores[(s_id, arg_id)] = role_probs
                    arg_role_srl_data[(s_id, arg_id)] = [srl_role_prediction, start_idx, end_idx]
            id_to_args[s_id] = args
        if len(arg_role_scores.keys()) != 0:
            data[process] = [sent_to_id, id_to_args, arg_role_scores, arg_role_srl_data,sentences]
    return data


def dump_ilp_json(data, ilp_data, ilp_scores, ilp_out_path):
    """Dump json file using the dictionary created from ilp data"""
    j_dump_data = []
    for process in data.keys():
        # list of sentences
        sent_list = []
        sent_to_id, id_to_args, arg_role_scores, arg_role_srl_data = data[process]
        ilp_score = ilp_scores[process]
        for sentence_text, s_id in sent_to_id.iteritems():
            # list of args
            arg_list = []
            for arg_id, arg_text in id_to_args[s_id]:
                srl_role_prediction, start_idx, end_idx = arg_role_srl_data[(s_id, arg_id)]
                # list of probs
                role_probs = map(lambda x: dict([x]), ilp_score[(s_id, arg_id)].items())
                ilp_r_vals = ilp_data[process][s_id][arg_id]
                ilp_i_vals =  {v: k for k, v in ilp_r_vals.items()}
                if 1 in ilp_i_vals:
                    ilp_role = ilp_config.roles[ilp_i_vals[1]]
                else:
                    ilp_role = "NONE"
                arg_list.append({'argId': arg_id,
                                 'text': arg_text,
                                 'rolePredicted': ilp_role,
                                 'startIdx': start_idx,
                                 'endIdx': end_idx,
                                 'probRoles': role_probs})
            sent_list.append({'sentenceId': s_id,
                              'text': sentence_text,
                              'predictionArgumentSpan': arg_list})
        j_dump_data.append({'process': process,
                            'sentences': sent_list})
    with open(ilp_out_path, 'w') as fp:
            json.dump(j_dump_data, fp, indent=4)


# Some useful utilities
def get_sentences(p_data):
    sent_to_id, id_to_args, arg_role_scores = p_data
    return [(v, k) for k, v in sent_to_id.iteritems()]


def get_sentence_args(sentence, p_data):
    sent_to_id, id_to_args, arg_role_scores = p_data
    s_id = sent_to_id[sentence]
    return id_to_args[s_id]


def get_role_scores(sentence, arg_id, role, p_data):
    sent_to_id, id_to_args, arg_role_scores = p_data
    s_id = sent_to_id[sentence]
    return arg_role_scores[s_id, arg_id][role]


def get_role_score_dict(p_data):
    sentences = get_sentences(p_data)
    roles = ilp_config.roles
    role_score_vars = {}
    for s_id, sentence in sentences:
        args = get_sentence_args(sentence, p_data)
        for a_id, arg in args:
            for r_id, role in enumerate(roles):
                role_score = get_role_scores(sentence, a_id, role, p_data)
                role_score_vars[s_id, a_id, r_id] = role_score
    return role_score_vars


def get_similarity_score(arg1, arg2):
    """"Call entailment function by passing args in both directions and return
    the best score."""
    ret = entailment.get_ai2_textual_entailment(arg1, arg2)
    a_scores = map(lambda x: x['score'], ret['alignments'])
    if len(a_scores):
        mean_a_score = np.mean(a_scores)
    else:
        mean_a_score = 0

    confidence = ret['confidence'] if ret['confidence'] else 0
    score1 = mean_a_score * confidence

    ret = entailment.get_ai2_textual_entailment(arg2, arg1)
    a_scores = map(lambda x: x['score'], ret['alignments'])
    if len(a_scores):
        mean_a_score = np.mean(a_scores)
    else:
        mean_a_score = 0

    confidence = ret['confidence'] if ret['confidence'] else 0
    score2 = mean_a_score * confidence
    return float(max(score1, score2))


def get_ilp_assignment_from_file(process):
    f = open(join(ilp_config.project_dir,'output', process+'_ilp.sol'))
    lines = f.readlines()
    data = filter(lambda x: x.startswith('Z'), lines)
    data =  map(lambda x: x[:-1], data)

    output_map = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for d in data:
        var, ind = d.split(" ")
        var_ids = var.split("_")
        s = int(var_ids[1])
        a = int(var_ids[2])
        r = int(var_ids[3])
        output_map[s][a][r] = int(ind)
    return output_map


def get_ilp_scores(process, srl_data, sim_data):
    """Use ILP assignments, insert it back into objective function and calculate
    the ILP score for a given assignement."""
    _, id_to_args, _, _ = srl_data[process]

    output_map = get_ilp_assignment_from_file(process)
    role_score_vals = get_role_score_dict(srl_data[process][:3])

    ilp_scores = defaultdict(lambda: defaultdict(float))

    for s_1, val_1 in output_map.iteritems():
        args_1 = id_to_args[s_1]
        for a_1, aval_1 in val_1.iteritems():
            arg_1 = dict(args_1)[a_1]
            for r, rv_1 in aval_1.iteritems():
                tmp = 0
                for s_2, val_2 in output_map.iteritems():
                    if s_1 != s_2:
                        args_2 = id_to_args[s_2]
                        for a_2, aval_2 in val_2.iteritems():
                            arg_2 = dict(args_2)[a_2]
                            rv_2 = aval_2[r]
                            tmp += rv_2 * sim_data[(arg_1, arg_2)]
                ilp_scores[s_1, a_1][ilp_config.roles[r]] = (float(role_score_vals[s_1, a_1, r]) * ilp_config.lambda_1) + (ilp_config.lambda_2 * tmp)
    return ilp_scores


def normalize_ilp_scores(ilp_scores):
    """Normalize the ilp scores."""
    norm_ilp_scores = {}
    for s_a_id, a_data in ilp_scores.iteritems():
        denom = sum(a_data.values())
        norm_vals = dict(map(lambda x: (x[0], x[1]/denom), a_data.items()))
        norm_ilp_scores[s_a_id]= norm_vals
    return norm_ilp_scores


def get_gold_data(d_gold):
    """Parse the gold data into python dictionary."""
    gold_data_raw = defaultdict(list)
    for process_dict in d_gold:
        process = process_dict['process']
        # list of sentences
        for sentence_dict in process_dict['sentences']:
            sent_id = sentence_dict['sentenceId']
            # list of arguments
            for arg_dict in sentence_dict['annotatedArgumentSpan']:
                start_id = int(arg_dict['startIdx'])
                end_id = int(arg_dict['endIdx'])
                role_type = arg_dict['annotatedRole']
                role_label = int(arg_dict['annotatedLabel'])
                argument=arg_dict['text']
                gold_data_raw[(sent_id,argument, start_id, end_id,process)].append((role_type, role_label))
    #print "PROCESS DICT",process_dict
    #print "SENTENCE DICT",sentence_dict
    #print "RAW",gold_data_raw
    

    gold_data = {}
    for k, v in gold_data_raw.iteritems():
        roles = []
        labels = []
        argument=[]
        for x in v:
            roles.append(x[0])
            labels.append(x[1])
            #argument.append(x[2])
        # if any role name has 1 as its value, set that (the first one) as the
        # gold role.
        if 1 in labels:
            gold_data[k] = roles[labels.index(1)]
        # if none of the roles have 1 value (but instead have -1), the set the
        # role of such argument span as 'NONE'
        elif np.sum(labels) == -4:
            gold_data[k] = 'NONE'
    return gold_data


def get_prediction_data(d_predict):
    """Parse the prediction data into python dictionary."""
    srl_data = defaultdict()
    for process_dict in d_predict:
        process = process_dict['process']
        # list of sentences
        for sentence_dict in process_dict['sentences']:
            sent_id = sentence_dict['sentenceId']
            # list of arguments
            for arg_dict in sentence_dict['predictionArgumentSpan']:
                start_id = int(arg_dict['startIdx'])
                end_id = int(arg_dict['endIdx'])
                role_predicted = arg_dict['rolePredicted']
                # create a dictionary with role label as key and predction score
                # of the role as value
                role_probs = {}
                for role_prob in arg_dict['probRoles']:
                    role_probs.update(role_prob)
                srl_data[(sent_id, start_id, end_id)] = (role_predicted, role_probs[role_predicted])
    return srl_data


def plot_precision_yield(plot_data, name='prec_recall', role=None):
    srl_plot_df, ilp_plot_df, semafor_plot_df, easysrl_plot_df = plot_data
    srl_plot_df = srl_plot_df.iloc[10:]
    ilp_plot_df = ilp_plot_df.iloc[10:]
    semafor_plot_df = semafor_plot_df.iloc[10:]
    easysrl_plot_df = easysrl_plot_df.iloc[10:]

    # plot size
    plt.rc('figure', figsize=(18,12))

    # plot lines
    plt.plot(srl_plot_df.index, srl_plot_df.precision, label=r'\textbf{SRL}', linewidth=3)
    plt.plot(ilp_plot_df.index, ilp_plot_df.precision, 'r--',label=r'\textbf{ILP}', linewidth=3)
    plt.plot(semafor_plot_df.index, semafor_plot_df.precision, 'g--',label=r'\textbf{SEMAFOR}', linewidth=3)
    plt.plot(easysrl_plot_df.index, easysrl_plot_df.precision, 'm--',label=r'\textbf{EasySRL}', linewidth=3)

    # configure plot
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.xlabel(r'\textbf{Recall}', fontsize=50)
    plt.ylabel(r'\textbf{Precison}', fontsize=50)
    plt.xlim([0, 1])
    plt.ylim([0, 1.005])
    plt.legend(loc='lower right', handlelength=3, prop={'size':45}) #borderpad=1.5, labelspacing=1.5,
    plt.tight_layout()

    # save plot
    if role:
        f_name = join(ilp_config.plots_dir, str(role) + "_" + str(name) + ".pdf")
    else:
        f_name = join(ilp_config.plots_dir, str(name) + ".pdf")
    plt.savefig(f_name)
    plt.close()


def plot_precision_yield_axes(plot_data, ax, fold, role=None):
    srl_plot_df, ilp_plot_df, semafor_plot_df, easysrl_plot_df = plot_data
    srl_plot_df = srl_plot_df.iloc[5:]
    ilp_plot_df = ilp_plot_df.iloc[5:]
    semafor_plot_df = semafor_plot_df.iloc[5:]
    easysrl_plot_df = easysrl_plot_df.iloc[5:]

    # plot size
    plt.rc('figure', figsize=(18,14))

    # plot lines
    ax.plot(srl_plot_df.index, srl_plot_df.precision, label=r'\textbf{SRL}', linewidth=3)
    ax.plot(ilp_plot_df.index, ilp_plot_df.precision, 'r--',label=r'\textbf{ILP}', linewidth=3)
    ax.plot(semafor_plot_df.index, semafor_plot_df.precision, 'g--',label=r'\textbf{SEMAFOR}', linewidth=3)
    ax.plot(easysrl_plot_df.index, easysrl_plot_df.precision, 'm--',label=r'\textbf{EasySRL}', linewidth=3)

    # configure plot
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel(r'\textbf{Recall}', fontsize=28)
    ax.set_ylabel(r'\textbf{Precison}', fontsize=28)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.005])
    if role:
        ax.set_title(str(role).title() + ' Fold ' + str(fold), fontsize=20)
    else:
        ax.set_title('Fold ' + str(fold), fontsize=20)
    ax.legend(loc='lower right', handlelength=3, prop={'size':15}) #borderpad=1.5, labelspacing=1.5,
    plt.tight_layout()


def plot_role_plot(data, name='prec_recall_folds', role=None):
    srl_plot_data, ilp_plot_data, semafor_plot_data, easysrl_plot_data = data

    # create 5 subplots for 5 roles in one column
    fig, ((ax1), (ax2), (ax3), (ax4), (ax5)) = plt.subplots(nrows=5, ncols=1, figsize=(10, 40))

    # call plot function on each of the 5 subplot axes
    plot_precision_yield_axes((srl_plot_data[1], ilp_plot_data[1], semafor_plot_data[1], easysrl_plot_data[1]), ax1, "1", role)
    plot_precision_yield_axes((srl_plot_data[2], ilp_plot_data[2] , semafor_plot_data[2], easysrl_plot_data[2]), ax2, "2", role)
    plot_precision_yield_axes((srl_plot_data[3], ilp_plot_data[3] , semafor_plot_data[3], easysrl_plot_data[3]), ax3, "3", role)
    plot_precision_yield_axes((srl_plot_data[4], ilp_plot_data[4] , semafor_plot_data[4], easysrl_plot_data[4]), ax4, "4", role)
    plot_precision_yield_axes((srl_plot_data[5], ilp_plot_data[5] , semafor_plot_data[5], easysrl_plot_data[5]), ax5, "5", role)

    # adjust spacing betwen plots
    fig.subplots_adjust(hspace=.3, wspace=-0.2)

    # save plot
    if role:
        f_name = join(ilp_config.plots_dir, str(role) + "_" + str(name) + ".pdf")
    else:
        f_name = join(ilp_config.plots_dir, str(name) + ".pdf")
    fig.savefig(f_name)
    plt.close(fig)


def plot_confusion_matrix(c_matrix, fold=None, filename=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(c_matrix, annot=True,  fmt='',
                xticklabels=labels, yticklabels=ilp_config.labels,
                linewidths=1, square=True);
    ax.xaxis.set_ticks_position("bottom")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    if fold:
        plt.title('Fold ' + str(fold) +' Confusion Matrix', fontsize=20);
    else:
        plt.title('Confusion Matrix', fontsize=20);
    if filename:
        f_name = join(ilp_config.plots_dir, str(filename) + ".pdf")
        plt.savefig(f_name)
    plt.close()


def subplot_confusion_matrix(c_matrix, ax, fold):
    sns.heatmap(c_matrix, annot=True,  fmt='',
                xticklabels=ilp_config.labels, yticklabels=ilp_config.labels,
                linewidths=1, square=True, ax=ax, cbar=False);
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(ilp_config.labels, rotation=30, fontsize=20)
    ax.set_yticklabels(ilp_config.labels[::-1], rotation=30, fontsize=20)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    ax.set_title("Fold " + str(fold) + ' Confusion Matrix', fontsize=20);


def plot_confusion_subplots(c_matrices, name="confusion_matrix"):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    subplot_confusion_matrix(c_matrices[1], ax1, "1")
    subplot_confusion_matrix(c_matrices[2], ax2, "2")
    subplot_confusion_matrix(c_matrices[3], ax3, "3")
    subplot_confusion_matrix(c_matrices[4], ax4, "4")
    #subplot_confusion_matrix(c_matrices[5], ax5, "5")
    fig.subplots_adjust(hspace=.8, wspace=-0.2)
    ax6.axis('off')
    f_name = join(ilp_config.plots_dir, str(name) + ".pdf")
    fig.savefig(f_name)
    plt.close(fig)


def generate_tex_table(df, name='accuracy_table'):
    filename = join(ilp_config.plots_dir, str(name) + ".tex")
    pdffile = join(ilp_config.plots_dir, str(name) + ".pdf")
    outname = join(ilp_config.plots_dir, str(name) + ".png")

    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    with open(filename, 'wb') as f:
        f.write(template.format(df.to_latex()))

    FNULL = open(os.devnull, 'w')
    subprocess.call(['pdflatex',  '-output-directory', ilp_config.plots_dir , filename], stdout=FNULL, stderr=subprocess.STDOUT)
    subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname], stdout=FNULL, stderr=subprocess.STDOUT)
