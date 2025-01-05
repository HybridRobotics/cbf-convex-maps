function [til] = plot_figure(robots, logs, a_hcbf, dist_margin)
    % Set figure position.
    % NOTE: Manually set the figure size in pts.
    fig = figure();
    % Create tiledlayout.
    til = tiledlayout(fig, 10, 2); % (rows, columns).
    til.TileSpacing = "tight";
    til.Padding = "tight";
    til.TileIndexing = "rowmajor"; % (default).

    %% First tile - Initial snapshot.
    nrobots = numel(robots);
    nT = logs.nT;
    ax = nexttile([4, 1]);
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

    %% Second tile - Final snapshot + trajectories.
    ax = nexttile([4, 1]);
    cl1_ = colormap(ax, "hot");
    cl2_ = colormap(ax, "cool");
    idx_seq = floor(linspace(1, 256, numel(robots)));
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
    % Plot trajectories.
    % hold on
    % for i = 1:nrobots
    %     hold on
    %     if 2*i <= nrobots
    %         cl_ = cl1_(idx_seq(i), :);
    %         plot3(ax, logs.x_seq{i}(1,:), logs.x_seq{i}(2,:), logs.x_seq{i}(3,:), ...
    %         '-', 'Color', cl_, 'LineWidth', 1.5);
    %     else
    %         cl_ = cl2_(idx_seq(i - nrobots/2), :);
    %         plot3(ax, logs.x_seq{i}(1,:), logs.x_seq{i}(2,:), logs.x_seq{i}(3,:), ...
    %         '-', 'Color', cl_, 'LineWidth', 1.5);
    %     end
    %     plot3(ax, logs.x_seq{i}(1,1), logs.x_seq{i}(2,1), logs.x_seq{i}(3,1), ...
    %         'o', 'Color', cl_);
    %     plot3(ax, logs.x_seq{i}(1,end), logs.x_seq{i}(2,end), logs.x_seq{i}(3,end), ...
    %         'x', 'Color', cl_);
    % end
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

    %% Third tile - Minimum distance.
    ax = nexttile([3, 1]);
    dist2_ode_seq = zeros(nrobots * (nrobots - 1) / 2, nT);
    dist2_opt_seq = zeros(nrobots * (nrobots - 1) / 2, nT);
    idx_ = 1;
    for i = 1:nrobots
        for j = i+1:nrobots
            dist2_ode_seq(idx_, :) = logs.dist2_seq{i, j};
            dist2_opt_seq(idx_, :) = logs.opt_dist2_seq{i, j};
            idx_ = idx_ + 1;
        end
    end
    dist2_opt_seq_min = squeeze(min(dist2_opt_seq, [], 1));
    dist2_opt_seq_max = squeeze(max(dist2_opt_seq, [], 1));
    dist2_ode_seq_min = squeeze(min(dist2_ode_seq, [], 1));
    dist2_ode_seq_max = squeeze(max(dist2_ode_seq, [], 1));
    skip_ = 100;
    % Plot distances.
    hold on
    % fill(ax, [logs.t_seq(1:skip_:end), fliplr(logs.t_seq(1:skip_:end))], ...
    %     [dist2_opt_seq_min(1:skip_:end), fliplr(dist2_opt_seq_max(1:skip_:end))], ...
    %     'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    fill(ax, [logs.t_seq(1:skip_:end), fliplr(logs.t_seq(1:skip_:end))], ...
        [dist2_ode_seq_min(1:skip_:end), fliplr(dist2_ode_seq_max(1:skip_:end))], ...
        'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, logs.t_seq, squeeze(mean(dist2_ode_seq, 1)), '-b');
    % plot(ax, logs.t_seq, squeeze(mean(dist2_opt_seq, 1)), '-.r');
    % Axis properties.
    ax.XLim = [0, logs.Tf];
    ax.YLim = [1e-4, 1e3];
    ax.Box = "off";
    ax.XTick = 0:10:logs.Tf;
    % ax.XTickLabel = [];
    ax.XTickLabel = {'0', ' ', '20', ' ', '40', ' ', '60', ' '};
    set(ax, 'YScale', 'log');
    ax.YTick = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3];
    ax.YTickLabel = {' ', ' ', '10^{-2}', ' ', '1', ' ', '10^{2}', ' '};
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "distance $(m^2)$", Interpreter = "latex");
    % Legends.
    legends = {"$h(t)$", "$h^*(t)$"};
    child_ = flipud(ax.Children);
    leg = legend([child_(2)], legends(1));
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "southeast";
    leg.Interpreter = "latex";

    %% Fourth tile - KKT error.
    ax = nexttile([3, 1]);
    zopt_ode_seq = zeros(nrobots * (nrobots - 1) / 2, 8, nT);
    zopt_opt_seq = zeros(nrobots * (nrobots - 1) / 2, 8, nT);
    yopt_ode_seq = zeros(nrobots * (nrobots - 1) / 2, 2, nT);
    yopt_opt_seq = zeros(nrobots * (nrobots - 1) / 2, 2, nT);
    idx_ = 1;
    for i = 1:nrobots
        for j = i+1:nrobots
            zopt_ode_seq(idx_, :, :) = logs.zopt_seq{i, j};
            zopt_opt_seq(idx_, :, :) = logs.opt_zopt_seq{i, j};
            yopt_ode_seq(idx_, :, :) = logs.yopt_seq{i, j};
            yopt_opt_seq(idx_, :, :) = logs.opt_yopt_seq{i, j};
            idx_ = idx_ + 1;
        end
    end
    zopt_err_seq = vecnorm(zopt_opt_seq - zopt_ode_seq, 2, 2);
    yopt_err_seq = vecnorm(yopt_opt_seq - yopt_ode_seq, 2, 2);
    zopt_opt_norm_seq = vecnorm(zopt_opt_seq, 2, 2);
    yopt_opt_norm_seq = vecnorm(yopt_opt_seq, 2, 2);
    zopt_rel_err_seq = zopt_err_seq ./ zopt_opt_norm_seq;
    yopt_rel_err_seq = yopt_err_seq ./ yopt_opt_norm_seq;
    skip_ = 100;
    zopt_rel_err_seq_min = 1e-10 + squeeze(min(zopt_rel_err_seq, [], 1))';
    zopt_rel_err_seq_max = 1e-10 + squeeze(max(zopt_rel_err_seq, [], 1))';
    yopt_rel_err_seq_min = 1e-10 + squeeze(min(yopt_rel_err_seq, [], 1))';
    yopt_rel_err_seq_max = 1e-10 + squeeze(max(yopt_rel_err_seq, [], 1))';
    hold on
    fill(ax, [logs.t_seq(1:skip_:end), fliplr(logs.t_seq(1:skip_:end))], ...
        [yopt_rel_err_seq_min(1:skip_:end), fliplr(yopt_rel_err_seq_max(1:skip_:end))], ...
        'r', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    fill(ax, [logs.t_seq(1:skip_:end), fliplr(logs.t_seq(1:skip_:end))], ...
        [zopt_rel_err_seq_min(1:skip_:end), fliplr(zopt_rel_err_seq_max(1:skip_:end))], ...
        'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, logs.t_seq, squeeze(mean(zopt_rel_err_seq, 1)), '-b');
    plot(ax, logs.t_seq, squeeze(mean(yopt_rel_err_seq, 1)), '-.r');
    % Axis properties.
    ax.XLim = [0, logs.Tf];
    ax.YLim = [1e-7, 5e-1];
    ax.Box = "off";
    ax.XTick = 0:10:logs.Tf;
    % ax.XTickLabel = [];
    ax.XTickLabel = {'0', ' ', '20', ' ', '40', ' ', '60', ' '};
    ax.YTick = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    ax.YTickLabel = {' ', ' ', '10^{-5}', ' ', '10^{-3}', ' ', '10^{-1}'};
    set(ax, 'YScale', 'log');
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "rel-err $()$", Interpreter = "latex");
    % Legends.
    legends = {"$|\Delta z^*(t)|/|z^*(t)|$", ...
        "$$|\Delta \lambda^*(t)|/|\lambda^*(t)|$"};
    child_ = flipud(ax.Children);
    leg = legend([child_(3), child_(4)], legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "southwest";
    leg.Interpreter = "latex";
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');

    %% Fifth tile - Distance error.
    ax = nexttile([3, 1]);
    skip_ = 100;
    dist_rel_err_seq = abs(sqrt(dist2_opt_seq) - sqrt(dist2_ode_seq)) ./ sqrt(dist2_opt_seq);
    dist_rel_err_seq_min = 1e-10 + squeeze(min(dist_rel_err_seq, [], 1));
    dist_rel_err_seq_max = 1e-10 + squeeze(max(dist_rel_err_seq, [], 1));
    hold on
    fill(ax, [logs.t_seq(1:skip_:end), fliplr(logs.t_seq(1:skip_:end))], ...
        [dist_rel_err_seq_min(1:skip_:end), fliplr(dist_rel_err_seq_max(1:skip_:end))], ...
        'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, logs.t_seq, squeeze(mean(dist_rel_err_seq, 1)), '-b');
    % Axis properties.
    ax.XLim = [0, logs.Tf];
    ax.XTick = 0:10:logs.Tf;
    ax.XTickLabel = {'0', ' ', '20', ' ', '40', ' ', '60', ' '};
    ax.YLim = [1e-7, 3e-1];
    ax.Box = "off";
    set(ax, 'YScale', 'log');
    ax.YTick = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    ax.YTickLabel = {' ', ' ', '10^{-5}', ' ', '10^{-3}', ' ', '10^{-1}'};
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "rel-err $()$", Interpreter = "latex");
    % Legends.
    legends = {"$\Delta h(t)/h(t)$"};
    child_ = flipud(ax.Children);
    leg = legend(child_(2), legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "northeast";
    leg.Interpreter = "latex";
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');

    %% Sixth tile - Distance derivative error.
    ax = nexttile([3, 1]);
    dist_opt_seq_ = sqrt(dist2_opt_seq(:, 1:end-1));
    dist_ode_seq_ = sqrt(dist2_ode_seq(:, 1:end-1));
    Ddist_err_seq = logs.opt_Ddist2_seq ./ (2*dist_opt_seq_) - ...
        logs.Ddist2_seq ./ (2*dist_ode_seq_);
    Ddist_err_seq_min = squeeze(min(Ddist_err_seq, [], 1));
    Ddist_err_seq_max = squeeze(max(Ddist_err_seq, [], 1));
    skip_ = 100;
    % Plot errors.
    hold on
    fill(ax, [logs.t_seq(1:skip_:end-1), fliplr(logs.t_seq(1:skip_:end-1))], ...
        [Ddist_err_seq_min(1:skip_:end), fliplr(Ddist_err_seq_max(1:skip_:end))], ...
        'b', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(logs.t_seq(1:end-1), zeros(1, length(logs.t_seq)-1), '--k');
    plot(logs.t_seq(1:end-1), mean(Ddist_err_seq, 1), '-b');
    % Axis properties.
    ax.XLim = [0, logs.Tf];
    ax.XTick = 0:10:logs.Tf;
    ax.XTickLabel = {'0', ' ', '20', ' ', '40', ' ', '60', ' '};
    % ax.YLim = [-3e-2, 3e-2];
    ax.Box = "off";
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "error $(m/s)$", Interpreter = "latex");
    % Legends.
    legends = {"$\Delta \dot{h}(t)$"};
    child_ = flipud(ax.Children);
    leg = legend(child_(3), legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "southeast";
    leg.Interpreter = "latex";
end
