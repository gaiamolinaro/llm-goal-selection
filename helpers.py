import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from scipy.stats import entropy
from itertools import permutations

pd.options.mode.chained_assignment = None  # default='warn'
import warnings

# ======================================================================================================================
# UTILITIES
# ======================================================================================================================


def adjust_color_brightness(hex_color, factor=0.6):
    """
    Adjust the brightness of a hex color.
    factor: 0-1 darkens the color, >1 lightens it
    """
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:], 16)
    
    # Adjust the brightness
    r = int(min(255, r * factor))
    g = int(min(255, g * factor))
    b = int(min(255, b * factor))
    
    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'

def get_family(model):
    family = ""
    if model.startswith("gpt-"):
        family = "openai"
    elif model.startswith("gemini"):
        family = "google"
    elif model.startswith("claude"):
        family = "anthropic"
    elif model.startswith("centaur"):
        family = "centaur"
    elif model.startswith("qwen"):
        family = "qwen"
    return family

def count_pattern_grouped(df, group_cols, sequence_col, pattern):
    m = len(pattern)

    def count_matches(x):
        x = np.array(x)
        if len(x) < m:
            return 0
        # Create rolling windows of size m
        windows = np.lib.stride_tricks.sliding_window_view(x, m)
        # Compare to pattern and count matches
        return (windows == pattern).all(axis=1).sum()

    counts = df.groupby(group_cols)[sequence_col].apply(count_matches).reset_index(name='pattern_count')
    return counts.pattern_count.fillna(0)

# ======================================================================================================================
# PLOTTING
# ======================================================================================================================

class Plotter():
    '''
        Data plotter 
    '''
    def __init__(self, task_df, train_df, learn_df, test_df,
                 n_learn_goals=6,
                 goal_colors=["#29a9b8", "#f58752", "#ab65ff", "#cccccc", "#553380", "#747474"],
                 position_colors=["#eCBe2C", "#79c445", "#ff5bb9", "#0f3687", "#c61821", "#e3d2C0"],
                 plotting_mode="paper"):

        self.task_df = task_df
        self.train_df = train_df
        self.learn_df = learn_df
        self.test_df = test_df

        self.n_learn_goals = n_learn_goals

        self.goal_colors=goal_colors
        self.position_colors=position_colors
        self.plotting_mode = plotting_mode
        self._set_plot_params()

        self.stretch_names = {0: "training phase", 1: "training and learning phase", 2: "learning phase",
            3: "learning and testing phase", 4: "whole experiment", 5: "testing phase"}

        self.n_subjs = learn_df.subj_id.nunique()
        self.n_conditions = learn_df.condition.nunique()
        self.conditions = np.sort(learn_df.condition.unique())
        self.n_learn_trials = learn_df.trial.nunique()
        self.n_train_trials = train_df.trial.nunique()
        self.n_test_trials = test_df.trial.nunique()

    def _set_plot_params(self):
        '''
        Set parameters for plotting
        '''
        if self.plotting_mode == "paper":
            self.font_size = 14
            self.k_color = "k"
            self.bg_color = "w"
            plt.rcParams["errorbar.capsize"] = 5
        elif self.plotting_mode == "poster":
            self.font_size = 20
            self.k_color = "w"
            self.bg_color = "#333333"
            plt.rcParams["errorbar.capsize"] = 10
        elif self.plotting_mode == "presentation":
            self.font_size = 20
            self.k_color = "k"
            self.bg_color = "w"
            plt.rcParams["errorbar.capsize"] = 10

        self.legend_font_size = self.font_size-2
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = self.font_size

        plt.rcParams["lines.color"] = self.k_color
        plt.rcParams["lines.markeredgecolor"] = self.k_color
        plt.rcParams["lines.markerfacecolor"] = self.k_color
        plt.rcParams["axes.edgecolor"] = self.k_color
        plt.rcParams["text.color"] = self.k_color
        plt.rcParams["axes.labelcolor"] = self.k_color
        plt.rcParams["xtick.color"] = self.k_color
        plt.rcParams["ytick.color"] = self.k_color
        plt.rcParams["lines.color"] = self.k_color
        plt.rcParams["axes.facecolor"] = self.bg_color
        plt.rcParams["figure.facecolor"] = self.bg_color

        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.labelspacing"] = 0
        plt.rcParams["legend.columnspacing"] = 0.7
        plt.rcParams["legend.handletextpad"] = 0.1

        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.default"] = "regular"
        plt.rcParams["mathtext.rm"] = "Arial"
        plt.rcParams["mathtext.it"] = "Arial:italic"
        plt.rcParams["mathtext.bf"] = "Arial:bold"
        self.fade = 0.4


    def _get_df_from_stretch(self, stretch):
        assert stretch in np.arange(6), "`stretch` must be an integer between 0 and 5"
        if stretch == 0:
            df = self.train_df.copy()
        elif stretch == 1:
            df = pd.concat([self.train_df, self.learn_df]).sort_values(by=["subj_num", "block", "trial"])
        elif stretch == 2:
            df = self.learn_df.copy()
        elif stretch == 3:
            df = pd.concat([self.learn_df, self.test_df]).sort_values(by=["subj_num", "block", "trial"])
        elif stretch == 4:
            df = self.task_df.copy()
        elif stretch == 5:
            df = self.test_df.copy()
        return df
    


    def _get_action_idx_df(self, by_position=True, stretch=1):

        self._set_plot_params()
        var = "choice" if by_position else "choice_id"
        df = self._get_df_from_stretch(stretch)
        
        actions_by_trial = df.query("event_id == 'action_selection'").groupby(["subj_num","trial", "block"])[var].apply(list).reset_index()
        actions_by_trial.rename(columns={var: 'action_list'}, inplace=True)

        feedback_df = df.query("event_id == 'feedback'")[['subj_num', 'trial', 'correct']]
        goal_df = df.query("event_id == 'goal_selection'")[['subj_num', 'trial', 'chosen_goal', 'chosen_goal_position']]
        condition_df = df[['subj_num', 'trial', 'condition']].drop_duplicates()

        df = pd.merge(actions_by_trial, feedback_df, on=['subj_num', 'trial'], how='left')
        df = pd.merge(df, goal_df, on=['subj_num', 'trial'], how='left')
        df = pd.merge(df, condition_df, on=['subj_num', 'trial'], how='left')

        all_actions = [list(p) for p in permutations(np.arange(4), 2)]
        if not by_position:
            all_actions += [list(p) for p in permutations(np.arange(4, 8), 2)]
        all_actions += [list(p) for p in permutations(np.arange(4), 4)]
        if not by_position:
            all_actions += [list(p) for p in permutations(np.arange(4, 8), 4)]
        df['chosen_action_index'] = df['action_list'].apply(lambda x: all_actions.index(x) if x in all_actions else np.nan)
        df['prev_chosen_action_index'] = df.groupby(["subj_num", "chosen_goal"])["chosen_action_index"].shift(1)
        return df
    

    # ==================================================================================================================
    # Metrics
    # ==================================================================================================================
    def get_metrics(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            # Initialize the metrics dataframe
            # Sort by subj_num so that subsequent positional (`.values`) assignments
            # from `df.groupby("subj_num")...` line up with the correct rows.
            init_cols = ["exp_id", "source", "subj_id", "subj_num", "condition"]
            self.metrics = (self.task_df.groupby('subj_id')[init_cols]
                            .first()
                            .sort_values(by="subj_num")
                            .reset_index(drop=True))
            # Initialize variables
            df = self.task_df.query("event_id == 'feedback'").reset_index(drop=True)
            n_subjs = df.subj_num.nunique()
            blocks = df.block.unique()
            n_blocks = df.block.nunique()
            learn_blocks = df.query("phase==1").block.unique()
            n_learn_blocks = df.query("phase==1").block.nunique()
            n_learn_goals = 6
            df["chosen_goal_cat"] = pd.Categorical(df.chosen_goal, categories=np.arange(n_learn_goals))
            gs_learn_df = df.query("phase==1").reset_index(drop=True)

            # Performance metrics
            # proportion of correct trials in the training, learning, and testing phases
            self.metrics[[f"p_correct_b{block_num}" for block_num in blocks]] = df.groupby(["subj_num", "block"]).correct.mean().values.reshape(n_subjs, n_blocks)
            self.metrics[f"p_correct_b{blocks[-1]}_base"] = df.query("block == @blocks[-1] & chosen_goal < @n_learn_goals").groupby(["subj_num"]).correct.mean().values 
            self.metrics[f"p_correct_b{blocks[-1]}_infer"] = df.query("block == @blocks[-1] & chosen_goal >= @n_learn_goals").groupby(["subj_num"]).correct.mean().values 
            self.metrics["p_correct_learning"] = df.query("block in @learn_blocks").groupby(["subj_num"]).correct.mean().values

            # Goal selection patterns (learning phase)
            # proportion chosen goals of a certain type
            p_goal = gs_learn_df.groupby(["subj_num", "block"]).chosen_goal_cat.value_counts(normalize=True).sort_index().values.reshape(n_subjs, n_learn_blocks, n_learn_goals)
            # 2-step goals
            self.metrics["p_2act_goals"] = p_goal[:, :, [0, 1, 4, 5]].sum(axis=2).mean(axis=1)
            # proportion of repeated goals
            gs_learn_df["repeat_goal_1back"] = gs_learn_df.chosen_goal.eq(gs_learn_df.chosen_goal.shift()).astype(int)
            self.metrics = self.metrics.merge(gs_learn_df.groupby(["subj_num"]).repeat_goal_1back.mean().reset_index().rename(columns={"repeat_goal_1back": "p_repeat_goal_1back"}), on="subj_num", how="left")
            # entropy
            self.metrics["goal_entropy"] = entropy(p_goal.mean(axis=1), axis=1) # average entropy in the entire task
            # number of times the 0-6 pattern was found without interruptions, over any permutation of the six goals
            self.metrics["n_max_any_0_6_pattern" ] = np.max([count_pattern_grouped(gs_learn_df, "subj_num", "chosen_goal_position", permutation) for permutation in permutations(np.array([0, 1, 2, 3, 4, 5]))], axis=0)
            # favorite (most chosen frequently) goal
            self.metrics["fav_goal"] = (gs_learn_df.groupby(["subj_num"], as_index=False)
                    .chosen_goal_cat
                    .value_counts(normalize=True)  # these are already sorted, with the highest-proprition goal at the top
                    .groupby(["subj_num"]).chosen_goal_cat.first() # so we pick the first chosen_goal
                    .astype(int)
                    .values
                    )
            self.metrics["fav_goal_position"] = (gs_learn_df.groupby(["subj_num"], as_index=False)
                    .chosen_goal_position
                    .value_counts(normalize=True)  # these are already sorted, with the highest-proprition goal at the top
                    .groupby(["subj_num"]).chosen_goal_position.first() # so we pick the first chosen_goal
                    .astype(int)
                    .values
                    )

            # Action selection patterns (learning phase): systematic hypothesis testing
            # (trying out ingredient-position permutations in a fixed order)
            action_index_df = self._get_action_idx_df()
            action_index_df["systematic_hypothesis_testing"] = (action_index_df.subj_num == action_index_df.subj_num.shift()) & (action_index_df.chosen_goal == action_index_df.chosen_goal.shift()) & (action_index_df["chosen_action_index"] - action_index_df["prev_chosen_action_index"] == 1).astype(int)
            action_index_df_sum = action_index_df.groupby(["subj_num"]).systematic_hypothesis_testing.sum().reset_index()
            self.metrics = self.metrics.merge(action_index_df_sum.rename(columns={"systematic_hypothesis_testing": "n_systematic_hypothesis_testing"}), how='left', on='subj_num')

            return


    # ==================================================================================================================
    # Plots by individuals/conditions
    # ==================================================================================================================

    def plot_trial_action_seq_indiv(self, ax, subj_idx=None, by_position=True, stretch=1, with_title=False, with_ylabel=True, with_xlabel=True, 
                                    color_code="correct", same_color="#00BCE3", correct_color="#00cc00", 
                                    legend_subj_idx=0, bbox_to_anchor=(0.85,1.1), legend_fontsize=None):

        self._set_plot_params()
        if legend_fontsize is None:
            legend_fontsize = self.legend_font_size
        var = "choice" if by_position else "choice_id"
        var_label = "Action position" if by_position else "Action"
        df = self._get_df_from_stretch(stretch)
        stretch_name = self.stretch_names[stretch]

        actions_by_trial = df.query("event_id == 'action_selection'").groupby(["subj_id", "trial"])[var].apply(list).reset_index()
        actions_by_trial.rename(columns={var: 'action_list'}, inplace=True)

        feedback_df = df.query("event_id == 'feedback'")[['subj_id', 'trial', 'correct']]
        goal_df = df.query("event_id == 'goal_selection'")[['subj_id', 'trial', 'chosen_goal', 'chosen_goal_position']]
        condition_df = df[['subj_id', 'trial', 'condition']].drop_duplicates()

        df_plot = pd.merge(actions_by_trial, feedback_df, on=['subj_id', 'trial'], how='left')
        df_plot = pd.merge(df_plot, goal_df, on=['subj_id', 'trial'], how='left')
        df_plot = pd.merge(df_plot, condition_df, on=['subj_id', 'trial'], how='left')
        
        chosen_actions = df_plot['action_list']
        
        all_actions = [list(p) for p in permutations(np.arange(4), 2)]
        if not by_position:
            all_actions += [list(p) for p in permutations(np.arange(4, 8), 2)]
        all_actions += [list(p) for p in permutations(np.arange(4), 4)]
        if not by_position:
            all_actions += [list(p) for p in permutations(np.arange(4, 8), 4)]
        chosen_action_indices = chosen_actions.apply(lambda x: all_actions.index(x) if x in all_actions else np.nan)

        correct = df_plot['correct']
        chosen_goal = df_plot['chosen_goal']

        phase = df.query("event_id == 'goal_selection'").reset_index(drop=True).phase
        phase_changes = np.where(phase.ne(phase.shift(1)))[0]-0.5

        if color_code == "correct":
            alt_col = correct_color
            fillcolor = np.where(correct, alt_col, "k")
            edgecolor = fillcolor
            legend_title = f"Correct {var_label.lower()}" 
            legend_labels = ["Yes", "No"] 
        elif color_code == "same":
            alt_col = same_color
            fillcolor = np.where(chosen_actions.eq(chosen_actions.shift()), alt_col, "k")
            edgecolor = fillcolor
            legend_title = f"{var_label} relative to previous"
            legend_labels = ["Same", "Different"]
        elif color_code == "chosen_goal_position":
            chosen_goal = df_plot['chosen_goal_position'] # Plot the chosen goal's position, not the chosen goal itself
            colors_dict = {i: self.position_colors[i] for i in range(len(self.position_colors))}
            colors_dict[np.nan] = self.bg_color
            edgecolor = chosen_goal.replace(colors_dict)
            fillcolor = np.where(correct, edgecolor, "none")
            legend_title = "Chosen goal position/Feedback"
        elif color_code == "chosen_goal":
            colors_dict = {i: self.goal_colors[i] for i in range(len(self.goal_colors))}
            colors_dict[np.nan] = self.bg_color
            edgecolor = chosen_goal.replace(colors_dict)
            fillcolor = np.where(correct, edgecolor, "none")
            legend_title = "Chosen goal/Feedback"
        else:
            raise ValueError("color_code should be one of ['correct', 'same', 'chosen_goal_position', 'chosen_goal']")
        
        if len(phase_changes) > 1:
            for change_idx in phase_changes[1:]:
                ax.axvline(x=change_idx, color='black', linestyle='--', alpha=1, zorder=0)
        ax.scatter(np.arange(len(chosen_action_indices)), chosen_action_indices, c=fillcolor, edgecolors=edgecolor)
        
        n_plotted_trials = len(df_plot)
        trials_per_block = self._get_df_from_stretch(1).trial_in_block.max()
        
        y_ticks = np.arange(len(all_actions)) if with_ylabel else []
        y_labels = [[int(a) for a in action] for action in all_actions] if with_ylabel else []
        ax.set_yticks(y_ticks, y_labels, fontsize=self.legend_font_size-2) 
        ax.set_ylabel(f"{var_label} sequence" if with_ylabel else "")
        ax.set_ylim(-1, len(all_actions))

        x = np.arange(n_plotted_trials)
        x_ticks = np.arange(0, np.max(x) + 1, trials_per_block) if with_xlabel else []
        x_labels = np.arange(0, np.max(x) + 1, trials_per_block) + 1 if with_xlabel else []
        ax.set_xticks(x_ticks, x_labels)
        ax.set_xlabel(f"Trial number\n({stretch_name})" if with_xlabel else "")
        
        condition = df_plot['condition'].iloc[0] if not df_plot.empty else ''
        ax.set_title(f"#{subj_idx+1}, c={condition}" if with_title else "", loc="left")

        if subj_idx == legend_subj_idx:
            if color_code in ["correct", "same"]:
                legend_elements = [Line2D([0], [0], marker="o", color="none", label=legend_labels[0], markerfacecolor=alt_col, markersize=8), 
                                Line2D([0], [0], marker="o", color="none", label=legend_labels[1], markerfacecolor=self.k_color, markersize=8)]
                ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=bbox_to_anchor, fontsize=self.legend_font_size, title=legend_title)
            elif color_code == "chosen_goal_position":
                legend_elements = [Line2D([0], [0], marker="o", color="none", markersize=8, markerfacecolor=c, markeredgecolor=c, label=c_idx+1) for c_idx, c in enumerate(self.position_colors)] + \
                        [Line2D([0], [0], marker="o", color="none", label="Correct", markerfacecolor=self.k_color, markeredgecolor=self.k_color, markersize=8), 
                        Line2D([0], [0], marker="o", color="none", label="Incorrect", markerfacecolor="none", markeredgecolor=self.k_color, markersize=8)]
                ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=bbox_to_anchor, ncol=len(legend_elements)//3, fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)
            else:
                legend_elements = [Line2D([0], [0], marker="o", color="none", markersize=8, markerfacecolor=c, markeredgecolor=c, label=c_idx+1) for c_idx, c in enumerate(self.goal_colors)] + \
                        [Line2D([0], [0], marker="o", color="none", label="Correct", markerfacecolor=self.k_color, markeredgecolor=self.k_color, markersize=8), 
                        Line2D([0], [0], marker="o", color="none", label="Incorrect", markerfacecolor="none", markeredgecolor=self.k_color, markersize=8)]
                ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=bbox_to_anchor, ncol=len(legend_elements)//3, fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)

        return ax

        

    def plot_trial_goal_seq_indiv(self, ax, subj_idx=None, by_position=True, stretch=1, with_title=False, with_ylabel=True, with_xlabel=True, 
                                    legend_subj_idx=0, bbox_to_anchor=(0.85,1.1), legend_fontsize=None):

        self._set_plot_params()
        if legend_fontsize is None:
            legend_fontsize = self.legend_font_size
        df = self._get_df_from_stretch(stretch)
        stretch_name = self.stretch_names[stretch]

        correct = df.query("event_id == 'feedback'")['correct']
        df = df.query("event_id == 'goal_selection'").reset_index(drop=True)
        
        chosen_goal = df['chosen_goal_position'] if by_position else df['chosen_goal']
        var_label = "Chosen goal position" if by_position else "Chosen goal"
        color_code = "chosen_goal_position" if by_position else "chosen_goal"

        phase = df.query("event_id == 'goal_selection'").reset_index(drop=True).phase
        phase_changes = np.where(phase.ne(phase.shift(1)))[0]-0.5

        if color_code == "chosen_goal_position":
            colors_dict = {i: self.position_colors[i] for i in range(len(self.position_colors))}
            colors_dict[np.nan] = self.bg_color
            edgecolor = chosen_goal.replace(colors_dict)
            fillcolor = np.where(correct, edgecolor, "none")
            legend_title = "Chosen goal position/Feedback"
        else:
            colors_dict = {i: self.goal_colors[i] for i in range(len(self.goal_colors))}
            colors_dict[np.nan] = self.bg_color
            edgecolor = chosen_goal.replace(colors_dict)
            fillcolor = np.where(correct, edgecolor, "none")
            legend_title = "Chosen goal/Feedback"

        if len(phase_changes) > 1:
            for change_idx in phase_changes[1:]:
                ax.axvline(x=change_idx, color='black', linestyle='--', alpha=1, zorder=0)
        ax.plot(np.arange(len(chosen_goal)), chosen_goal, 'k--', alpha=0.3)
        ax.scatter(np.arange(len(chosen_goal)), chosen_goal, c=fillcolor, edgecolors=edgecolor, zorder=3)

        n_plotted_trials = len(df)
        trials_per_block = self._get_df_from_stretch(1).trial_in_block.max()
        
        y_ticks = np.arange(self.n_learn_goals) if with_ylabel else []
        y_tick_labels = y_ticks + 1 if with_ylabel else []
        ax.set_yticks(y_ticks, y_tick_labels, fontsize=self.legend_font_size-2) 
        ax.set_ylabel(f"{var_label}" if with_ylabel else "")
        ax.set_ylim(-1, self.n_learn_goals)

        x = np.arange(n_plotted_trials)
        x_ticks = np.arange(0, np.max(x) + 1, trials_per_block) if with_xlabel else []
        x_labels = np.arange(0, np.max(x) + 1, trials_per_block) + 1 if with_xlabel else []
        ax.set_xticks(x_ticks, x_labels)
        ax.set_xlabel(f"Trial number\n({stretch_name})" if with_xlabel else "")
        
        condition = df['condition'].iloc[0]
        ax.set_title(f"#{subj_idx+1}, c={condition}" if with_title else "", loc="left")

        if subj_idx == legend_subj_idx:
            if color_code == "chosen_goal_position":
                legend_elements = [Line2D([0], [0], marker="o", color="none", markersize=8, markerfacecolor=c, markeredgecolor=c, label=c_idx+1) for c_idx, c in enumerate(self.position_colors)] + \
                        [Line2D([0], [0], marker="o", color="none", label="Correct", markerfacecolor=self.k_color, markeredgecolor=self.k_color, markersize=8), 
                        Line2D([0], [0], marker="o", color="none", label="Incorrect", markerfacecolor="none", markeredgecolor=self.k_color, markersize=8)]
                ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=bbox_to_anchor, ncol=len(legend_elements)//3, fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)
            else:
                legend_elements = [Line2D([0], [0], marker="o", color="none", markersize=8, markerfacecolor=c, markeredgecolor=c, label=c_idx+1) for c_idx, c in enumerate(self.goal_colors)] + \
                        [Line2D([0], [0], marker="o", color="none", label="Correct", markerfacecolor=self.k_color, markeredgecolor=self.k_color, markersize=8), 
                        Line2D([0], [0], marker="o", color="none", label="Incorrect", markerfacecolor="none", markeredgecolor=self.k_color, markersize=8)]
                ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=bbox_to_anchor, ncol=len(legend_elements)//3, fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)

        return ax


def compute_closest_human_distances(plotters, models, prefixes,
                                    distance_metric="euclidean"):
    """Compute min distance from each subject to the closest human, on normalized features.

    Each group's features are z-scored using the human values (mean/std of the
    human distribution) before computing distances.

    Returns (model_min_dists, model_min_locs dicts).
    """
    from scipy.spatial.distance import cdist

    def _build_feature_matrix(plotter, prefixes):
        cols = []
        for metric in prefixes:
            vals = plotter.metrics[metric].values.astype(float)
            cols.append(vals)
        return np.column_stack(cols)

    # Build raw matrices for every group
    raw_mats = {}
    raw_mats["human"] = _build_feature_matrix(plotters["human"], prefixes)
    for model in models:
        raw_mats[model] = _build_feature_matrix(plotters[model], prefixes)

    # Each group is z-scored by the human values
    human_mean = np.nanmean(raw_mats["human"], axis=0)
    human_std = np.nanstd(raw_mats["human"], axis=0)
    human_std[human_std == 0] = 1
    def _normalize(mat):
        return (mat - human_mean) / human_std

    human_mat = _normalize(raw_mats["human"])

    # Human-to-human distances (excluding self)
    human_dists = cdist(human_mat, human_mat, metric=distance_metric)
    np.fill_diagonal(human_dists, np.inf)
    human_min_locs = np.argmin(human_dists, axis=1)
    human_min_dists = human_dists[np.arange(len(human_dists)), human_min_locs]

    # Model-to-human distances
    model_min_locs = {}
    model_min_dists = {}
    for model in models:
        model_mat = _normalize(raw_mats[model])
        dists = cdist(model_mat, human_mat, metric=distance_metric)
        model_min_locs[model] = np.argmin(dists, axis=1)
        model_min_dists[model] = dists[np.arange(len(dists)), model_min_locs[model]]
    model_min_dists["human"] = human_min_dists
    model_min_locs["human"] = human_min_locs
    return model_min_dists, model_min_locs


def plot_closest_human_distances(min_dists, models,
                                 model_names_dict, model_colors_dict, model_markers_dict,
                                 distance_metric="euclidean", figsize=(10, 5)):
    """Plot bar + scatter of min distances to closest human for each agent.

    Returns (fig, ax).
    """
    import seaborn as sns

    # Build long-form DataFrame
    records = []
    for val in min_dists["human"]:
        records.append({"Agent": model_names_dict["human"], "Min. distance": val})
    for model in models:
        for val in min_dists[model]:
            records.append({"Agent": model_names_dict[model], "Min. distance": val})
    swarm_df = pd.DataFrame(records)

    agent_order = [model_names_dict["human"]] + [model_names_dict[m] for m in models]
    all_keys = ["human"] + list(models)
    palette = {model_names_dict[k]: (model_colors_dict[k] if k != "human" else "k") for k in all_keys}

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=swarm_df, x="Agent", y="Min. distance", order=agent_order,
                palette=palette, errorbar="se", capsize=0.15, ax=ax,
                err_kws={"linewidth": 1.5, "color": "k", "zorder": 100})

    for i, key in enumerate(all_keys):
        agent_name = model_names_dict[key]
        vals = swarm_df.loc[swarm_df["Agent"] == agent_name, "Min. distance"].values
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(vals))
        marker = model_markers_dict[key]
        color = model_colors_dict[key] if key != "human" else "k"
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   marker=marker, c="white", edgecolors=color, linewidths=0.5,
                   s=50, alpha=1.0, zorder=3)

    ax.set_xticks(range(len(agent_order)))
    ax.set_xticklabels(agent_order, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(f"Min. {distance_metric.capitalize()} distance\nto closest human", fontsize=14)
    sns.despine(ax=ax)
    fig.tight_layout()

    return fig, ax


def print_closest_human_stats(min_dists, models, model_names_dict):
    """Print summary table and MWU tests for closest-human distances."""
    print(f"{'Agent':<25} {'Median':>8} {'Mean':>8} {'SD':>8}")
    print("-" * 55)
    print(f"{'Humans (excl. self)':<25} {np.median(min_dists['human']):>8.2f} {np.mean(min_dists['human']):>8.2f} {np.std(min_dists['human']):>8.2f}")
    for model in models:
        d = min_dists[model]
        print(f"{model_names_dict[model]:<25} {np.median(d):>8.2f} {np.mean(d):>8.2f} {np.std(d):>8.2f}")

    print(f"\n{'Model vs. Humans':<25} {'U':>10} {'p (MWU)':>12} {'Direction':>12}")
    print("-" * 65)
    for model in models:
        d = min_dists[model]
        U, p_mwu = stats.mannwhitneyu(d, min_dists['human'], alternative="two-sided")
        direction = "higher" if np.median(d) > np.median(min_dists['human']) else "lower"
        p_str = "< 0.001" if p_mwu < 0.001 else f"{p_mwu:.3f}"
        print(f"{model_names_dict[model]:<25} {U:>10.1f} {p_str:>12} {direction:>12}")


def fmt_summary(vals):
    """Format mean ± std (median) for an array of values."""
    v = vals[~np.isnan(vals)]
    return f"{np.mean(v):.2f} ± {np.std(v):.2f} ({np.median(v):.2f})"


def build_comparison_tables(plotters, models, prefixes, cat_vars,
                            model_names_dict, include_human_descriptives=False):
    """Build descriptive and statistical comparison tables (each model vs human).

    Returns (desc_df, stats_df).
    """
    display_names = [model_names_dict[m] for m in models]
    human_name = model_names_dict["human"]
    metric_names = list(prefixes.values())

    desc_data = {}
    stats_data = {}

    for metric, metric_name in prefixes.items():
        human_vals = plotters["human"].metrics[metric].values.astype(float)
        if include_human_descriptives:
            desc_data.setdefault(human_name, {})[metric_name] = fmt_summary(human_vals)

        for model, display_name in zip(models, display_names):
            model_vals = plotters[model].metrics[metric].values.astype(float)
            model_clean = model_vals[~np.isnan(model_vals)]
            human_clean = human_vals[~np.isnan(human_vals)]

            # MWU test (median comparison)
            U, p_mwu = stats.mannwhitneyu(model_clean, human_clean)
            desc_data.setdefault(display_name, {})[metric_name] = fmt_summary(model_vals)
            stats_data.setdefault((display_name, "Md."), {})[metric_name] = f"{U:.1f}{stars(p_mwu)}"

            # Distribution test
            if metric in cat_vars:
                all_cats = set(model_clean.astype(int)) | set(human_clean.astype(int))
                counts_model = np.array([np.sum(model_clean == c) for c in sorted(all_cats)])
                counts_human = np.array([np.sum(human_clean == c) for c in sorted(all_cats)])
                contingency = np.array([counts_model, counts_human])
                chi2, p_dist, dof, expected = stats.chi2_contingency(contingency)
                stats_data.setdefault((display_name, "Dist."), {})[metric_name] = f"{chi2:.2f}{stars(p_dist)}"
            else:
                ks_stat, p_dist = stats.ks_2samp(model_clean, human_clean)
                stats_data.setdefault((display_name, "Dist."), {})[metric_name] = f"{ks_stat:.3f}{stars(p_dist)}"

    desc_df = pd.DataFrame(desc_data, index=metric_names)
    desc_df.columns.name = "Model"
    desc_df.index.name = "Metric"

    stats_df = pd.DataFrame(stats_data, index=metric_names)
    stats_df.columns = pd.MultiIndex.from_tuples(stats_df.columns)
    stats_df.index.name = "Metric"

    return desc_df, stats_df


def stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""
