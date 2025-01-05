function [til] = plot_figure(out, x1, C1, x_seq2, C2, Tf, timestamps)
    assert(C1.nz == 3);

    % Set figure position.
    % NOTE: Manually set the figure size in pts.
    fig = figure();
    % Create tiledlayout.
    til = tiledlayout(fig, 2, 2); % (rows, columns).
    til.TileSpacing = "tight";
    til.Padding = "tight";
    til.TileIndexing = "rowmajor"; % (default).

    %% First tile - Snapshot.
    ax = nexttile;
    % Plot trajectory.
    hold on
    plot3(ax, x_seq2(1, 1:Tf), x_seq2(2, 1:Tf), x_seq2(3, 1:Tf), ...
        "--k", LineWidth=1);
    plot3(ax, x_seq2(1, 1), x_seq2(2, 1), x_seq2(3, 1), "ok");
    plot3(ax, x_seq2(1, Tf), x_seq2(2, Tf), x_seq2(3, Tf), "ok");
    % Plot sets.
    C1.plot_surf(x1, ax, 'r', 1, 0.75);
    for i = 1:length(timestamps)
        t = timestamps(i);
        x2 = x_seq2(:, t);
        C2.plot_surf(x2, ax, "cyan", 1, 0.75);
    end
    % Axis properties.
    ax.XLim = [-10, 7];
    ax.YLim = [-9, 12];
    ax.ZLim = [-4, 5];
    ax.XTick = []; ax.YTick = []; ax.ZTick = [];
    ax.Box = "off";
    view(ax, [25, 18]);
    axis equal
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    % Legends.
    legends = {"$x(t)$", "$\mathcal{C}_1$", "$\mathcal{C}_2(x(t))$"};
    child_ = flipud(ax.Children);
    leg = legend([child_(1), child_(4), child_(5)], legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "northwest";
    leg.Interpreter = "latex";
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 1]');

    %% Second tile - KKT error.
    ax = nexttile;
    % Plot errors.
    zopt_err_seq = vecnorm(out.zopt_opt_seq - out.zopt_ode_seq, 2);
    yopt_err_seq = vecnorm(out.yopt_opt_seq - out.yopt_ode_seq, 2);
    hold on
    plot(ax, out.t_seq, zopt_err_seq ./ vecnorm(out.zopt_opt_seq, 2), '-b');
    plot(ax, out.t_seq, yopt_err_seq ./ vecnorm(out.yopt_opt_seq, 2), '-.r');
    % Axis properties.
    ax.YLim = [1e-5, 1e-1];
    ax.Box = "off";
    set(ax, 'YScale', 'log');
    % ax.XTickLabel = [];
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "rel-err $()$", Interpreter = "latex");
    % Legends.
    legends = {"$|\Delta z^*(t)|/|z^*(t)|$", ...
        "$$|\Delta \lambda^*(t)|/|\lambda^*(t)|$"};
    child_ = flipud(ax.Children);
    leg = legend(child_, legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "northeast";
    leg.Interpreter = "latex";
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');

    %% Third tile - Distance error.
    ax = nexttile;
    % Plot errors.
    dist_err_seq = sqrt(out.dist2_opt_seq) - sqrt(out.dist2_ode_seq);
    plot(out.t_seq, dist_err_seq ./ sqrt(out.dist2_opt_seq), '-b');
    % Axis properties.
    ax.YLim = [0, 4e-4];
    ax.Box = "off";
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "rel-err $()$", Interpreter = "latex");
    % Legends.
    legends = {"$\Delta h(t)/h(t)$"};
    child_ = flipud(ax.Children);
    leg = legend(child_, legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "northeast";
    leg.Interpreter = "latex";

    %% Fourth tile - Distance derivative error.
    ax = nexttile;
    % Plot errors.
    hold on
    plot(out.t_seq(1:end-1), zeros(1, length(out.t_seq)-1), '--k');
    Ddist_err_seq = out.Ddist2_opt_seq ./ (2*sqrt(out.dist2_opt_seq(1:end-1))) ...
        - out.Ddist2_analytic_seq ./ (2*sqrt(out.dist2_ode_seq(1:end-1)));
    plot(out.t_seq(1:end-1), Ddist_err_seq, '-b');
    % Axis properties.
    ax.YLim = [-3e-2, 3e-2];
    ax.Box = "off";
    % Label properties.
    ax.FontUnits = "points";
    ax.FontSize = 8;
    xlabel(ax, "$t$ $(s)$", Interpreter = "latex");
    ylabel(ax, "error $(m/s)$", Interpreter = "latex");
    % Legends.
    legends = {"$\Delta \dot{h}(t)$"};
    child_ = flipud(ax.Children);
    leg = legend(child_(2), legends);
    leg.FontSize = 8;
    leg.ItemTokenSize = [15, 18]; % Box size, default [30, 18], *Undocumented*.
    leg.NumColumns = 1;
    leg.Location = "northeast";
    leg.Interpreter = "latex";
    leg.BoxFace.ColorType = 'truecoloralpha';
    leg.BoxFace.ColorData = uint8(255 * [1, 1, 1, 0.5]');
end
