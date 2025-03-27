import matplotlib.pyplot as plot
import numpy as np
import plotly.graph_objects as go

class Visualizer:

    def __init__(self, tracing_model_solved, result_dict, N, K, beds, t_start, t_end, filename):
        # Store given values
        self.tracing_model_solved = tracing_model_solved
        self.result_dict = result_dict
        self.N = N
        self.N_total = sum(N)
        self.K = K
        self.beds = beds
        self.t_start = t_start
        self.t_end = t_end
        self.filename = filename

        # Sanity checks
        assert(self.t_start < self.t_end)

        # Unpack results
        self.t_vals = self.result_dict["t_vals"]
        self.S_vals = self.result_dict["S_vals"]
        self.S_vals_aggr = self._aggregate_over_groups(self.S_vals)
        self.E_vals = self.result_dict["E_vals"]
        self.E_vals_aggr = self._aggregate_over_groups(self.E_vals)
        self.I_asym_vals = self.result_dict["I_asym_vals"]
        self.I_asym_vals_aggr = self._aggregate_over_groups(self.I_asym_vals)
        self.I_sym_vals = self.result_dict["I_sym_vals"]
        self.I_sym_vals_aggr =  self._aggregate_over_groups(self.I_sym_vals)
        self.I_sev_vals = self.result_dict["I_sev_vals"]
        self.I_sev_vals_aggr =  self._aggregate_over_groups(self.I_sev_vals)
        self.R_vals = self.result_dict["R_vals"]
        self.R_vals_aggr = self._aggregate_over_groups(self.R_vals)
        self.D_vals = self.result_dict["D_vals"]
        self.D_vals_aggr = self._aggregate_over_groups(self.D_vals)
        self.sev_total_vals = self.result_dict["sev_total"]

        # Preparing lists of simulation results that are only used if a tracing model was solved
        if self.tracing_model_solved:
            self.E_tracked_vals = self.result_dict["E_tracked_vals"]
            self.E_tracked_vals_aggr = self._aggregate_over_groups(self.E_tracked_vals)
            self.Q_asym_vals = self.result_dict["Q_asym_vals"]
            self.Q_asym_vals_aggr = self._aggregate_over_groups(self.Q_asym_vals)
            self.Q_sym_vals = self.result_dict["Q_sym_vals"]
            self.Q_sym_vals_aggr = self._aggregate_over_groups(self.Q_sym_vals)
            self.Q_sev_vals = self.result_dict["Q_sev_vals"]
            self.Q_sev_vals_aggr = self._aggregate_over_groups(self.Q_sev_vals)

    def plot_all_curves(self):
        figure, axes = plot.subplots()
        figure.subplots_adjust(bottom = 0.15, left=0.15)
        axes.grid(linestyle = ':', linewidth = 0.5, color = "#808080")
        axes.set_xlabel("Days", fontsize=14)
        axes.set_ylabel("Individuals", fontsize=14)
        axes.tick_params(axis='both', which='major', labelsize=14)

        group_linestyles = {0: "solid", 1: "dotted"} # well, yes, that's hard-coded for K = 2

        for k in range(self.K):
            S_plot, = axes.plot(self.t_vals, self.S_vals[k],
                                color = "lightskyblue")#, linestyle = group_linestyles[k])
            S_plot.set_label("S_" + str(k))

            E_plot, = axes.plot(self.t_vals, self.E_vals[k],
                                color = "teal")#, linestyle = group_linestyles[k])
            E_plot.set_label("E_" + str(k))

            if self.tracing_model_solved:
                E_tracked_plot, = axes.plot(self.t_vals, self.E_tracked_vals[k],
                                            color = "deeppink")#, linestyle = group_linestyles[k])
                E_tracked_plot.set_label("E_tracked_" + str(k))

            I_asym_plot, = axes.plot(self.t_vals, self.I_asym_vals[k],
                                     color = "gold")#, linestyle = group_linestyles[k])
            I_asym_plot.set_label("I_asym_" + str(k))

            I_sym_plot, = axes.plot(self.t_vals, self.I_sym_vals[k],
                                    color = "orange")#, linestyle = group_linestyles[k])
            I_sym_plot.set_label("I_sym_" + str(k))

            I_sev_plot, = axes.plot(self.t_vals, self.I_sev_vals[k],
                                    color = "chocolate")#, linestyle = group_linestyles[k])
            I_sev_plot.set_label("I_sev_" + str(k))

            if self.tracing_model_solved:
                Q_asym_plot, = axes.plot(self.t_vals, self.Q_asym_vals[k],
                                         color = "plum")#, linestyle = group_linestyles[k])
                Q_asym_plot.set_label("Q_asym_" + str(k))

                Q_sym_plot, = axes.plot(self.t_vals, self.Q_sym_vals[k],
                                         color = "blueviolet")#, linestyle = group_linestyles[k])
                Q_sym_plot.set_label("Q_sym_" + str(k))

                Q_sev_plot, = axes.plot(self.t_vals, self.Q_sev_vals[k],
                                         color = "navy")#, linestyle = group_linestyles[k])
                Q_sev_plot.set_label("Q_sev_" + str(k))

            R_plot, = axes.plot(self.t_vals, self.R_vals[k],
                                color = "limegreen")#, linestyle = group_linestyles[k])
            R_plot.set_label("R_" + str(k))

            D_plot, = axes.plot(self.t_vals, self.D_vals[k],
                                color = "firebrick")#, linestyle = group_linestyles[k])
            D_plot.set_label("D_" + str(k))

        sev_total_plot, = axes.plot(self.t_vals, self.sev_total_vals,
                                    color = "chocolate", linestyle = "dashed")
        sev_total_plot.set_label("sev_total")

        # Plot horizontal lines for beds
        beds_plot = axes.hlines(float(self.beds), self.t_start, self.t_end,
                                color = "cornflowerblue", linestyle = "dashed")
        beds_plot.set_label("beds")

        # Shrink current axis by 20%
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plot.savefig(self.filename + "_full.png")
        ymax = 0.2 * self.N_total
        axes.set_ylim([-ymax * 0.05, ymax])
        plot.savefig(self.filename + "_small.png")
        ymax = 0.005 * self.N_total
        axes.set_ylim([-ymax * 0.05, ymax])
        plot.savefig(self.filename + "_super_small.png")
        plot.close("all")

    def plot_aggregated_curves(self, return_fig = False):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.t_vals, y=self.S_vals_aggr, mode='lines', name='Susceptible'))
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.E_vals_aggr, mode='lines', name='Exposed'))

        if self.tracing_model_solved:
            fig.add_trace(go.Scatter(x=self.t_vals, y=self.E_tracked_vals_aggr, mode='lines', name='E_tracked'))

        fig.add_trace(go.Scatter(x=self.t_vals, y=self.I_asym_vals_aggr, mode='lines', name='Asymptomatic Infected', line=dict(color='gold')))
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.I_sym_vals_aggr, mode='lines', name='Symptomatic Infected', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.I_sev_vals_aggr, mode='lines', name='Severe Infected', line=dict(color='chocolate')))

        if self.tracing_model_solved:
            fig.add_trace(go.Scatter(x=self.t_vals, y=self.Q_asym_vals_aggr, mode='lines', name='Q_asym', line=dict(color='plum')))
            fig.add_trace(go.Scatter(x=self.t_vals, y=self.Q_sym_vals_aggr, mode='lines', name='Q_sym', line=dict(color='blueviolet')))
            fig.add_trace(go.Scatter(x=self.t_vals, y=self.Q_sev_vals_aggr, mode='lines', name='Q_sev', line=dict(color='navy')))

        fig.add_trace(go.Scatter(x=self.t_vals, y=self.R_vals_aggr, mode='lines', name='Recovered', line=dict(color='limegreen')))
        fig.add_trace(go.Scatter(x=self.t_vals, y=self.D_vals_aggr, mode='lines', name='Deceased', line=dict(color='firebrick')))

        # Plot horizontal lines for beds
        fig.add_trace(go.Scatter(
            x=[self.t_start, self.t_end],
            y=[self.beds, self.beds],
            mode='lines',
            line=dict(color='cornflowerblue', dash='dash'),
            name='Beds'
        ))

        fig.update_layout(
            # title='Aggregated Curves',
            xaxis_title='Days',
            yaxis_title='Individuals',
            legend_title='Legend',
            font=dict(size=14),
            # legend=dict(
            #     yanchor="top",
            #     y=0.99,
            #     xanchor="right",
            #     x=0.01
            # )
        )
        
        if return_fig:
            return fig

        fig.write_image(self.filename + "_aggr_full.png")
        ymax = 0.2 * self.N_total
        fig.update_yaxes(range=[-ymax * 0.05, ymax])
        fig.write_image(self.filename + "_aggr_small.png")
        ymax = 0.005 * self.N_total
        fig.update_yaxes(range=[-ymax * 0.05, ymax])
        fig.write_image(self.filename + "_aggr_super_small.png")
        
    def paper_plot_figure_1(self):
        """
        Attention: This method contains code that is hard-coded for K=2
        """

        figure, axes = plot.subplots()
        figure.subplots_adjust(bottom = 0.15, left=0.15)
        axes.grid(linestyle = ':', linewidth = 0.5, color = "#808080")
        axes.set_xlabel("Days", fontsize=14)
        axes.set_ylabel("Individuals", fontsize=14)
        axes.tick_params(axis='both', which='major', labelsize=14)

        # 0: old; 1: young
        group_linestyles = {0: "solid", 1: "dotted"} # well, yes, that's hard-coded for K = 2

        # hard-coded for K = 2
        I_0_aggr_vals = [(x + y + z) for x, y, z in zip(self.I_asym_vals[0], self.I_sym_vals[0], self.I_sev_vals[0])]
        I_0_aggr_plot, = axes.plot(self.t_vals, I_0_aggr_vals, color = "tab:orange", linewidth=3.5, linestyle = group_linestyles[0])

        # hard-coded for K = 2
        I_1_aggr_vals = [(x + y + z) for x, y, z in zip(self.I_asym_vals[1], self.I_sym_vals[1], self.I_sev_vals[1])]
        I_1_aggr_plot, = axes.plot(self.t_vals, I_1_aggr_vals, color = "tab:orange", linewidth=3.5, linestyle = group_linestyles[1])

        for k in range(self.K):
            S_vals_k_abs = [self.S_vals[k][t] for t in range(len(self.S_vals[k]))]
            S_plot, = axes.plot(self.t_vals, S_vals_k_abs,
                                color = "tab:green", linewidth=3.5, linestyle = group_linestyles[k])
            #S_plot.set_label("S_" + str(k))

            R_vals_k_abs = [self.R_vals[k][t] for t in range(len(self.R_vals[k]))]
            R_plot, = axes.plot(self.t_vals, R_vals_k_abs,
                                color = "tab:blue", linewidth=3.5, linestyle = group_linestyles[k])
            #R_plot.set_label("R_" + str(k))

        plot.yticks(np.arange(0, 8e7, 1e7), ['0',
                                             r'$10\,$M',
                                             r'$20\,$M',
                                             r'$30\,$M',
                                             r'$40\,$M',
                                             r'$50\,$M',
                                             r'$60\,$M',
                                             r'$70\,$M'])
        plot.savefig(self.filename + "_figure_1.pdf")
        plot.close("all")

    def paper_plot_figure_3(self):
        figure, axes = plot.subplots()
        figure.subplots_adjust(bottom = 0.15, left=0.2)
        axes.grid(linestyle = ':', linewidth = 0.5, color = "#808080")
        axes.set_xlabel("Days", fontsize=14)
        axes.set_ylabel("Individuals", fontsize=14)
        axes.tick_params(axis='both', which='major', labelsize=14)

        # 0: old; 1: young
        group_linestyles = {0: "solid", 1: "dotted"} # well, yes, that's hard-coded for K = 2

        for k in range(self.K):

            I_and_Q_sym_total_vals_k = [(x + y) for x, y in zip(self.I_sym_vals[k], self.Q_sym_vals[k])]
            I_and_Q_sym_total_plot, = axes.plot(self.t_vals, I_and_Q_sym_total_vals_k,
                                                color = "tab:blue", linewidth=3.5, linestyle = group_linestyles[k])

            I_and_Q_asym_total_vals_k = [(x + y) for x, y in zip(self.I_asym_vals[k], self.Q_asym_vals[k])]
            I_and_Q_asym_total_plot, = axes.plot(self.t_vals, I_and_Q_asym_total_vals_k,
                                                 color = "tab:green", linewidth=3.5, linestyle = group_linestyles[k])

            I_and_Q_sev_total_vals_k = [(x + y) for x, y in zip(self.I_sev_vals[k], self.Q_sev_vals[k])]
            I_and_Q_sev_total_plot, = axes.plot(self.t_vals, I_and_Q_sev_total_vals_k,
                                                color = "tab:purple", linewidth=3.5, linestyle = group_linestyles[k])

            D_total_vals_k = [x for x in self.D_vals[k]]
            D_plot, = axes.plot(self.t_vals, D_total_vals_k, color = "tab:red", linewidth=3.5, linestyle = group_linestyles[k])

        I_and_Q_sev_total_vals = [(a + b + c + d) for a, b, c, d in zip(self.I_sev_vals[0],
                                                                                       self.I_sev_vals[1],
                                                                                       self.Q_sev_vals[0],
                                                                                       self.Q_sev_vals[1])]
        I_and_Q_sev_total_plot, = axes.plot(self.t_vals, I_and_Q_sev_total_vals, color = "tab:orange", linewidth=3.5, linestyle = "dashed")


        # Plot horizontal line for beds
        beds_plot = axes.hlines(self.beds, self.t_start, self.t_end, color = "black", linewidth=3.5, linestyle = "dashed")


        plot.yticks(np.arange(0, 400000, 50000), ['0',
                                                  r'$50\,$K',
                                                  r'$100\,$K',
                                                  r'$150\,$K',
                                                  r'$200\,$K',
                                                  r'$250\,$K',
                                                  r'$300\,$K',
                                                  r'$350\,$K'])
        ymax = 375000
        axes.set_ylim([-ymax * 0.05, ymax])
        plot.savefig(self.filename + "_figure_3.pdf")
        plot.close("all")

    def paper_plot_figure_4(self):
        figure, axes = plot.subplots()
        figure.subplots_adjust(bottom = 0.15, left=0.15)
        axes.grid(linestyle = ':', linewidth = 0.5, color = "#808080")
        axes.set_xlabel("Days", fontsize=14)
        axes.set_ylabel("Individuals", fontsize=14)
        axes.tick_params(axis='both', which='major', labelsize=14)

        # 0: old, 1: young
        group_linestyles = {0: "solid", 1: "dotted"} # well, yes, that's hard-coded for K = 2

        for k in range(self.K):

            I_and_Q_sev_total_vals_k = [(x + y) for x, y in zip(self.I_sev_vals[k], self.Q_sev_vals[k])]
            I_and_Q_sev_total_plot, = axes.plot(self.t_vals, I_and_Q_sev_total_vals_k,
                                                color = "tab:purple", linewidth=3.5, linestyle = group_linestyles[k])

            D_total_vals_k = [x for x in self.D_vals[k]]
            D_plot, = axes.plot(self.t_vals, D_total_vals_k, color = "tab:red", linewidth=3.5, linestyle = group_linestyles[k])

        I_and_Q_sev_total_vals = [(a + b + c + d) for a, b, c, d in zip(self.I_sev_vals[0],
                                                                                       self.I_sev_vals[1],
                                                                                       self.Q_sev_vals[0],
                                                                                       self.Q_sev_vals[1])]
        I_and_Q_sev_total_plot, = axes.plot(self.t_vals, I_and_Q_sev_total_vals, color = "tab:orange", linewidth=3.5, linestyle = "dashed")


        # Plot horizontal line for beds
        beds_plot = axes.hlines(self.beds, self.t_start, self.t_end, color = "black", linewidth=3.5, linestyle = "dashed")


        plot.yticks(np.arange(0, 120000, 20000), ['0',
                                                  r'$20\,$K',
                                                  r'$40\,$K',
                                                  r'$60\,$K',
                                                  r'$80\,$K',
                                                  r'$100\,$K'])
        ymax = 105000
        axes.set_ylim([-ymax * 0.05, ymax])
        plot.savefig(self.filename + "_figure_4.pdf")
        plot.close("all")

    def _aggregate_over_groups(self, list_of_lists):
        assert(len(list_of_lists) == self.K)
        aggregated_list = []
        for time_idx in range(len(self.t_vals)):
            aggregated_value = 0
            for group_idx in range(self.K):
                aggregated_value += list_of_lists[group_idx][time_idx]
            aggregated_list.append(aggregated_value)
        assert(len(self.t_vals) == len(aggregated_list))
        return aggregated_list

# class Visualizer
