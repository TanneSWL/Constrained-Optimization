% Exercise 2.4: Solve QP using quadprog and plot results

load('QP_Test.mat');  % Provides H, g, C, dl, du, l, u

% Rename to match quadprogs syntax
H = double(H);
f = double(g);

% Variable bounds
lb = l;
ub = u;

% Inequality constraints: dl <= C'x <= du
A = [ C'; -C' ];
b = [ du; -dl ];

% No equality constraints
Aeq = [];
beq = [];

% Solve QP
options = optimoptions('quadprog', 'Display', 'iter');
tic;
[x_opt, fval, exitflag, output] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);
solve_time = toc;

% Report results
disp("Optimal solution x_opt:");
disp(x_opt);
disp("Iterations:");
disp(output.iterations);
disp("Solver time (s):");
disp(solve_time);

% Plot solution using PlotSolutionQP
PlotSolutionQP(x_opt)

