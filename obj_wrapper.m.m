function score = obj_wrapper(tbl)
    % Unpack variables from table
    x = [tbl.eig_rho, tbl.W_in_a, tbl.alpha, tbl.log10_beta, tbl.k_scaled, tbl.kb];
    score = objective_double_arm(x);
end
