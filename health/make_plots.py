import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from quoll.utils import flatten_list, key_match_in_dicts, append_array
from quoll.metrics import closest_point
from quoll.draw import find_unique_label_position

print('making plots...')

work_dir = '/Volumes/slim/more_projects/japan_health/'
file = work_dir + 'FigureData_update.xlsx'
xl = pd.ExcelFile(file)

# settings
plot_1 = True
plot_1a = False
plot_1b = False
plot_1c = False
plot_1d = True
plot_2 = False
plot_3 = False
plot_4 = False  # requires variable from plot_1

if plot_1:

    # figure 1
    df1 = xl.parse('Fig1Data')
    data = df1['CarbonF'].values
    labels = df1['Category'].values

    cmap = plt.get_cmap('tab20c')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]

    plt.figure()
    ax = plt.subplot(111)

    plt.pie(data, colors=colors)
    ax.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=2)

    plt.savefig(work_dir + 'figs/fig1_pie.png', dpi=700, bbox_inches='tight')
    plt.savefig(work_dir + 'figs/fig1_pie.pdf', bbox_inches='tight')

    plt.close()

    if plot_1a:
        # figure 1a
        df1 = xl.parse('Fig1Data')
        sub_categories = df1['OuterLabel'].unique().tolist()

        for idx, cat in enumerate(sub_categories):

            selection = df1[df1['OuterLabel'] == cat]

            data = selection['CarbonF'].values
            labels = selection['Category'].values

            cmap = plt.get_cmap('tab20')
            colors = [cmap(i) for i in np.linspace(0, 1, len(data))]

            plt.figure()
            ax = plt.subplot(111)

            plt.pie(data, colors=colors)
            ax.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0), frameon=False, ncol=2)
            plt.title(cat)

            plt.savefig(work_dir + 'figs/fig1.' + str(idx) + '_pie.png', dpi=700, bbox_inches='tight')
            plt.savefig(work_dir + 'figs/fig1.' + str(idx) + '_pie.pdf', bbox_inches='tight')

            plt.clf()

        plt.close()

    if plot_1b:
        # figure 1b (nested pie)
        df1 = xl.parse('Fig1bData')

        #   group categories
        group_names = df1['NewOuterLabel'].unique().tolist()
        group_size = []
        for g in group_names:
            selection = df1[df1['NewOuterLabel'] == g]
            total = selection['CarbonF'].values.sum()
            group_size.append(total)

        subgroup_names = df1['Label'].tolist()
        subgroup_size = df1['CarbonF'].values

        #   Pie settings
        outside_radius = 1.0+(1/3)
        inner_radius = (2/3)*outside_radius
        explode_offset = 0
        explode_inner = []
        explode_outer = []

        for i in range(len(subgroup_names)):
            explode_inner.append(explode_offset/2)

        for i in range(len(group_names)):
            explode_outer.append(explode_offset)

        #   outer ring
        fig, ax = plt.subplots()
        ax.axis('equal')
        mypie, _ = ax.pie(group_size, radius=outside_radius, labels=group_names, textprops={'fontsize': 10}
                          , explode=explode_outer)
        plt.setp(mypie, width=outside_radius/3, edgecolor='white')

        #   inner ring
        mypie2, _ = ax.pie(subgroup_size, radius=inner_radius, labels=subgroup_names, labeldistance=0.8
                           , explode=explode_inner, textprops={'fontsize': 8})
        plt.setp(mypie2, width=inner_radius/2, edgecolor='white')

        plt.savefig(work_dir + 'figs/fig1_nested_pie.png', dpi=700, bbox_inches='tight')
        plt.close()

    if plot_1c:
        # figure 1c (flip rings around)
        df1 = xl.parse('Fig1cData')

        #   group categories
        group_names = df1['NewOuterLabel'].unique().tolist()
        group_size = []
        for g in group_names:
            selection = df1[df1['NewOuterLabel'] == g]
            total = selection['CarbonF'].values.sum()
            group_size.append(total)

        subgroup_names = df1['Label'].tolist()
        subgroup_size = df1['CarbonF'].values

        #   Pie settings
        outside_radius = 1.0+(1/3)
        inner_radius = (2/3)*outside_radius
        inner_ring_width = inner_radius/2
        outer_ring_width = outside_radius/3

        # colour maps
        colour_maps = ['Reds', 'Purples', 'Blues', 'Wistia', 'Oranges', 'Greens']
        assert len(group_names) <= len(colour_maps), 'Not enough colour maps'
        name_colour_store = []  # remember link between categories and plots for further plots

        discrete_inner_colours = []
        discrete_outer_colours = []
        for idx, g in enumerate(group_names):
            selection = df1[df1['NewOuterLabel'] == g]
            cmap = plt.get_cmap(colour_maps[idx])
            discrete_colors = [cmap(i) for i in np.linspace(0, 1, len(selection)+3)]
            reverse_colors = discrete_colors[::-1]
            a = reverse_colors[1]
            b = reverse_colors[2:-1]
            discrete_inner_colours.append(a)
            discrete_outer_colours.append(b)

            # remember colour matches for later (figure 4)
            names = selection['Category'].tolist()
            for i, c in enumerate(b):
                name_colour_store.append({'group': g, 'name': names[i], 'c': b[i]})

        discrete_outer_colours = flatten_list(discrete_outer_colours)

        # create figure
        fig, ax = plt.subplots()
        ax.axis('equal')

        #   outer ring
        outer_wedges, _ = ax.pie(subgroup_size, radius=outside_radius, colors=discrete_outer_colours)

        #   inner wedges
        inner_wedges, texts = ax.pie(group_size, radius=inner_radius, textprops={'fontsize': 10}
                                     , colors=discrete_inner_colours)  # , labels=group_names
        plt.setp(inner_wedges, width=inner_ring_width, edgecolor='white')

        #   label outer wedges
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center")

        for i, p in enumerate(outer_wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(subgroup_names[i], xy=(x, y), xytext=(1.8*np.sign(x), 1.8*y), horizontalalignment=horizontalalignment
                        , **kw)

        plt.setp(outer_wedges, width=outer_ring_width, edgecolor='white')

        # Legend for inner ring
        plt.legend(inner_wedges, group_names, loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=3)

        plt.savefig(work_dir + 'figs/fig1_nested_pie_3.png', dpi=400, bbox_inches='tight')
        plt.close()

    # figure 1d
    if plot_1d:

        df1 = xl.parse('Fig1eData')

        #   Inner group category labels
        inner_labels = []
        group_names = df1['NewOuterLabel'].unique().tolist()
        group_size = []
        for g in group_names:

            # Totals
            selection = df1[df1['NewOuterLabel'] == g]
            total = selection['CarbonF'].values.sum()
            group_size.append(total)

            # labels
            category_total = '{:.0f}'.format(df1[df1['NewOuterLabel'] == g]['CarbonF'].sum()/1000)
            inner_labels.append(str(g) + '; ' + category_total + 'Mt')

        subgroup_names = df1['Label'].tolist()
        subgroup_size = df1['CarbonF'].values

        #   Pie settings
        outside_radius = 1.0+(1/3)
        inner_radius = (2/3)*outside_radius
        inner_ring_width = inner_radius/2
        outer_ring_width = outside_radius/3

        # colour maps
        colour_maps = ['Reds', 'Purples', 'Blues', 'Wistia', 'Oranges', 'Greens']
        assert len(group_names) <= len(colour_maps), 'Not enough colour maps'
        name_colour_store = []  # remember link between categories and plots for further plots

        discrete_inner_colours = []
        discrete_outer_colours = []
        for idx, g in enumerate(group_names):
            selection = df1[df1['NewOuterLabel'] == g]
            cmap = plt.get_cmap(colour_maps[idx])
            discrete_colors = [cmap(i) for i in np.linspace(0, 1, len(selection)+3)]
            reverse_colors = discrete_colors[::-1]
            a = reverse_colors[1]
            b = reverse_colors[2:-1]
            discrete_inner_colours.append(a)
            discrete_outer_colours.append(b)

            # remember colour matches for later (figure 4)
            names = selection['Label'].tolist()
            for i, c in enumerate(b):
                name_colour_store.append({'group': g, 'name': names[i], 'c': b[i]})

        discrete_outer_colours = flatten_list(discrete_outer_colours)

        # create figure
        fig, ax = plt.subplots()
        ax.axis('equal')

        #   outer ring
        outer_wedges, _ = ax.pie(subgroup_size, radius=outside_radius, colors=discrete_outer_colours)

        #   inner wedges
        inner_wedges, texts = ax.pie(group_size, radius=inner_radius
                                     , textprops={'fontsize': 10, 'color': 'w', 'weight': 'bold'}
                                     , colors=discrete_inner_colours, labels=inner_labels
                                     , labeldistance=0.6*inner_radius)

        #   label outer wedges
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
        label_coords = None  # remember plotted label coordinates to avoid overlaps
        closest_label_pos = 0.13
        max_adjust_attempts = 10

        for i, p in enumerate(outer_wedges):

            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))

            x_txt = 1.8*np.sign(x)
            y_txt = 1.8*y

            if label_coords is not None and label_coords.ndim > 1 \
                    and closest_point(np.array([x_txt, y_txt]), label_coords) < closest_label_pos:
                x_txt, y_txt = find_unique_label_position(x_txt, y_txt, label_coords, closest_label_pos
                                                          , 0.025, alter=['y'], max_attempts=max_adjust_attempts)

            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})

            outer_label = df1[df1['Label'] == subgroup_names[i]]['NewOuterLabel'].values[0]
            subgroup_name_label = str(outer_label) + '; ' + subgroup_names[i]

            ax.annotate(subgroup_name_label, xy=(x, y), xytext=(x_txt, y_txt)
                        , horizontalalignment=horizontalalignment, **kw)

            label_coords = append_array([x_txt, y_txt], label_coords)

        plt.setp(outer_wedges, width=outer_ring_width, edgecolor='white')

        # Legend for inner ring
        plt.legend(inner_wedges, group_names, loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=4)

        plt.savefig(work_dir + 'figs/fig_1d_nested_pie.png', dpi=400, bbox_inches='tight')
        plt.close()

if plot_2:
    # figure 2
    df1 = xl.parse('Fig2Data')
    df_tr = df1.transpose()

    labels = df1['Disease short'].values
    series_labels = list(df1)

    series_1 = series_labels[-2]
    series_2 = series_labels[-1]
    series_2_label = series_2.replace('greater than ', '$>$')
    series_1_label = series_1.replace('less than ', '$\leq$')

    a = df1[series_1].values
    b = df1[series_2].values

    b = b + a

    a = a/1000
    b = b/1000

    x_pos = [i for i, _ in enumerate(labels)]  # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    plt.figure()
    ax = plt.subplot(111)

    p2 = plt.barh(x_pos, b, label=series_2_label, color='xkcd:salmon', edgecolor='xkcd:slate grey', linewidth=0.5)
    p1 = plt.barh(x_pos, a, label=series_1_label, color='xkcd:dark sky blue', edgecolor='xkcd:slate grey', linewidth=0.5)

    plt.xlabel('Carbon footprint (MtCO$_2$-e)', labelpad=10)
    plt.yticks(x_pos, labels)
    plt.grid(which='major', axis='x', linestyle='--')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2)

    plt.savefig(work_dir + 'figs/fig2_bar_chart.png', dpi=700, bbox_inches='tight')
    plt.savefig(work_dir + 'figs/fig2_bar_chart.pdf', bbox_inches='tight')

    plt.close()

if plot_3:
    # Figure 3
    df1 = xl.parse('Fig3Data')

    labels = df1['Disease short'].values
    series_labels = list(df1)

    series_1 = series_labels[-2]
    series_2 = series_labels[-1]

    a = df1[series_1].values
    b = df1[series_2].values

    a = a
    b = b

    x_pos = [i for i, _ in enumerate(labels)]  # the x locations for the groups
    bar_width = 0.35       # the width of the bars: can also be len(x) sequence
    x_pos_2 = [x + bar_width for x in x_pos]
    tick_positions = [r + 0.5*bar_width for r in range(len(labels))]

    plt.figure()
    ax = plt.subplot(111)

    p1 = plt.barh(x_pos, a, height=bar_width, label=series_1, color='xkcd:cool green'
                  , edgecolor='xkcd:slate grey', linewidth=0.3)
    p2 = plt.barh(x_pos_2, b, height=bar_width, label=series_2, color='xkcd:lilac'
                  , edgecolor='xkcd:slate grey', linewidth=0.3)

    plt.xlabel('Carbon footprint (tCO$_2$e/patient)', labelpad=10)
    plt.yticks(tick_positions, labels)
    # plt.legend(loc='upper right', frameon=False)
    plt.grid(which='major', axis='x', linestyle='--')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2)

    plt.savefig(work_dir + 'figs/fig3_bar_chart.png', dpi=700, bbox_inches='tight')
    plt.savefig(work_dir + 'figs/fig3_bar_chart.pdf', bbox_inches='tight')

    plt.close()

if plot_4:
    # Figure 4
    df1 = xl.parse('Fig4DataFlipped')
    scale = 10e6

    x_labels = df1['Year'].values
    series_labels = list(df1)
    last_category = series_labels[-1]
    del series_labels[-1], series_labels[0]

    x_pos = [i for i, _ in enumerate(x_labels)]  # the x locations for the groups

    total_array = None
    scatter_data = []
    width = 0.4

    clrs = plt.cm.get_cmap('tab20')

    plt.figure()
    ax = plt.subplot(111)
    last = None

    # Make stacked bar chart
    for idx, series in enumerate(series_labels):

        # select series
        data = df1[series].values
        data = data/scale

        plt.bar(x_pos, data, width, edgecolor='xkcd:slate grey', linewidth=0.5, label=series, bottom=last
                , color=plt.cm.Set3(idx))

        if last is None:
            last = data
        else:
            last = last + data

    plt.xticks(x_pos, x_labels)
    ax.set_ylabel('Induced GHG emissions per patient (MT CO$_2$-e)', labelpad=10)

    # Overlay scatter plot
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.scatter(x_pos, df1[last_category].values, label='Share of the national total emissions', alpha=0.6, c='red')
    ax2.set_ylabel('Share of national GHG emissions (%)', labelpad=10)
    ax2.set_ylim(4.3, 5)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), frameon=False, ncol=2)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.05, -0.17), frameon=False)

    plt.savefig(work_dir + 'figs/fig4_bar_chart.png', dpi=700, bbox_inches='tight')
    plt.savefig(work_dir + 'figs/fig4_bar_chart.pdf', bbox_inches='tight')

    plt.close()

    # Figure 4b (label top of bar chart)
    df1 = xl.parse('Fig4DataFlipped')
    scale = 1

    x_labels = df1['Year'].values
    series_labels = list(df1)
    last_category = series_labels[-1]
    del series_labels[-1], series_labels[0]

    x_pos = [i for i, _ in enumerate(x_labels)]  # the x locations for the groups

    total_array = None
    scatter_data = []
    width = 0.4

    clrs = plt.cm.get_cmap('tab20')

    plt.figure()
    ax = plt.subplot(111)
    last = None

    # Make stacked bar chart
    for idx, series in enumerate(series_labels):

        # select series
        data = df1[series].values
        data = data/scale

        plt.bar(x_pos, data, width, edgecolor='xkcd:slate grey', linewidth=0.5, label=series, bottom=last
                , color=plt.cm.Set3(idx))

        if last is None:
            last = data
        else:
            last = last + data

    plt.xticks(x_pos, x_labels)

    yearly_totals = df1[series_labels].sum(axis=1).values / scale
    plt.ylim(top=1.1*max(yearly_totals))

    ax.set_ylabel('Induced GHG emissions per patient (MT CO$_2$-e)', labelpad=10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), frameon=False, ncol=2)

    # Add share labels
    share_values = df1[last_category].values
    for idx, x in enumerate(x_labels):
        ax.text(idx-0.25, yearly_totals[idx] + .25, str(round(share_values[idx], 2)) + '%')

    plt.savefig(work_dir + 'figs/fig4b_bar_chart.png', dpi=400, bbox_inches='tight')
    plt.savefig(work_dir + 'figs/fig4b_bar_chart.pdf', bbox_inches='tight')

    plt.close()

    # Figure 4c (fix colours)
    df1 = xl.parse('Fig4DataFlipped')
    df2 = xl.parse('Fig4Data')
    scale = 1

    x_labels = df1['Year'].values
    series_labels = list(df1)
    last_category = series_labels[-1]
    del series_labels[-1], series_labels[0]

    x_pos = [i for i, _ in enumerate(x_labels)]  # the x locations for the groups

    total_array = None
    scatter_data = []
    width = 0.4

    # create figure
    plt.figure()
    ax = plt.subplot(111)
    last = None

    # Make stacked bar chart
    for idx, series in enumerate(series_labels):

        # select series
        data = df1[series].values
        data = data / scale

        # get the color
        colour_match = key_match_in_dicts(name_colour_store, 'name', series)
        if not colour_match == []:
            colour = colour_match[0]['c']
        elif series == series_labels[-1] and not colour_match:
            colour = 'xkcd:cool green'
        else:
            raise ValueError('Could not match series name to figure 1 names')

        # plot
        plt.bar(x_pos, data, width, edgecolor='xkcd:slate grey', linewidth=0.5, label=series, bottom=last
                , color=colour)

        # cumulative height
        if last is None:
            last = data
        else:
            last = last + data

    plt.xticks(x_pos, x_labels)

    yearly_totals = df1[series_labels].sum(axis=1).values / scale
    plt.ylim(top=1.1 * max(yearly_totals))

    ax.set_ylabel('Carbon footprint (MtCO$_2$e)', labelpad=10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.43), frameon=False, ncol=2)

    # Add share labels
    share_values = df1[last_category].values
    for idx, x in enumerate(x_labels):
        ax.text(idx-0.2, yearly_totals[idx] + 0.05*yearly_totals[idx], str(round(share_values[idx], 2)) + '%')

    plt.savefig(work_dir + 'figs/fig4c_bar_chart.png', dpi=700, bbox_inches='tight')

    plt.close()

print('finished')