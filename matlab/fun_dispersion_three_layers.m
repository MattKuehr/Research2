function f_vec = fun_dispersion_three_layers(k_vec,q)

global phi_A1 phi_A2 L_A1 L_A2 eps_A eps_F delta_A alpha_F beta_F L_F

k = k_vec(1,:) + 1i*k_vec(2,:);
L = L_A1 + L_A2 + L_F;

T_A1 = transfer_mat_A(k,L_A1,eps_A,delta_A,phi_A1);
T_A2 = transfer_mat_A(k,L_A2,eps_A,delta_A,phi_A2);
T_F = transfer_mat_F(k,L_F,eps_F,alpha_F,beta_F);

f = det( T_A1*T_F*T_A2 - exp(1i*q*L) * eye(4,4) );

f_vec = [ real(f); imag(f) ];

end