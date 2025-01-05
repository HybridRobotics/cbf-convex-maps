function [til] = plot_env(robots, logs)
    % Set figure position.
    % NOTE: Manually set the figure size in pts.
    fig = figure();
    % Create tiledlayout.
    til = tiledlayout(fig, 1, 10); % (rows, columns).
    til.TileSpacing = "tight";
    til.Padding = "tight";
    til.TileIndexing = "rowmajor"; % (default).

    %% First tile - Initial snapshot.
    nrobots = numel(robots);
    nT = logs.nT;
    ax = nexttile([1, 3]);
    % Plot surfaces.
    hold on
    for i = 1:nrobots
        hold on
        if 2*i <= nrobots
            cs_ = 'r';
        else
            cs_ = 'g';
        end
        robots{i}.sets{1}.plot_surf(logs.x_seq{i}(:, 1), ax, cs_, 1, 0.75);
    end
    % Axis properties.
    % ax.XLim = [-10, 7];
    % ax.YLim = [-9, 12];
    % ax.ZLim = [-4, 5];
    ax.XTick = []; ax.YTick = []; ax.ZTick = [];
    ax.Box = "off";
    view(ax, [22, 20]);
    axis equal
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    %
    child_1 = flipud(ax.Children);

    %% Second tile - Final snapshot + trajectories.
    ax = nexttile([1, 3]);
    % Plot surfaces.
    hold on
    for i = 1:nrobots
        hold on
        if 2*i <= nrobots
            cs_ = 'r';
        else
            cs_ = 'g';
        end
        robots{i}.sets{1}.plot_surf(logs.x_seq{i}(:, end), ax, cs_, 1, 0.75);
    end
    % Axis properties.
    % ax.XLim = [-10, 7];
    % ax.YLim = [-20, 25];
    % ax.ZLim = [-4, 5];
    ax.XTick = []; ax.YTick = []; ax.ZTick = [];
    ax.Box = "off";
    view(ax, [22, 20]);
    axis equal
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    % Legends.
    child_ = flipud(ax.Children);

    %% Second tile - Final snapshot + trajectories.
    ax = nexttile([1, 3]);
    cl1_ = colormap(ax, "hot");
    cl2_ = colormap(ax, "cool");
    idx_seq = floor(linspace(1, 256, numel(robots)));
    Plot trajectories.
    hold on
    for i = 1:nrobots
        hold on
        if 2*i <= nrobots
            cl_ = cl1_(idx_seq(i), :);
            plot3(ax, logs.x_seq{i}(1,:), logs.x_seq{i}(2,:), logs.x_seq{i}(3,:), ...
            '-', 'Color', cl_, 'LineWidth', 1.5);
        else
            cl_ = cl2_(idx_seq(i - nrobots/2), :);
            plot3(ax, logs.x_seq{i}(1,:), logs.x_seq{i}(2,:), logs.x_seq{i}(3,:), ...
            '-', 'Color', cl_, 'LineWidth', 1.5);
        end
        plot3(ax, logs.x_seq{i}(1,1), logs.x_seq{i}(2,1), logs.x_seq{i}(3,1), ...
            'o', 'Color', cl_);
        plot3(ax, logs.x_seq{i}(1,end), logs.x_seq{i}(2,end), logs.x_seq{i}(3,end), ...
            'x', 'Color', cl_);
    end
    % Axis properties.
    % ax.XLim = [-10, 7];
    % ax.YLim = [-20, 25];
    % ax.ZLim = [-4, 5];
    ax.XTick = []; ax.YTick = []; ax.ZTick = [];
    ax.Box = "off";
    view(ax, [22, 20]);
    axis equal
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    % Legends.
    legends = {"Swarm 1", "Swarm 2"};
    child_ = flipud(ax.Children);
    leg = legend([child_(1), child_(81)], legends); % Each surface has 10 plots.
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 15]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "southwest";
    leg.Interpreter = "latex";
    % legend off;
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');

    %% Legend.
    legends = gobjects(4, 1);
    colors = {'r', 'g', 'r', 'g'};
    names = {'Swarm 1', 'Swarm 2', ''};
    legends = {"Swarm 1", "Swarm 2"};
    child_ = flipud(ax.Children);
    leg = legend([child_(1), child_(81)], legends); % Each surface has 10 plots.
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 15]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "southwest";
    leg.Interpreter = "latex";
    % legend off;
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');
end
