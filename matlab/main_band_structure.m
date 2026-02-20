
global phi_A1 phi_A2 L_A1 L_A2 eps_A eps_F delta_A alpha_F beta_F L_F

L = 2*pi;
L_A1 = L/4; L_A2 = L/4; L_F = L/2;    % L_A1 = 0; L_A2 = 0; L_F = L;
eps_A = 7.0+1.0j; eps_F = 1.0;       % eps_A = 13.0+3.5i; eps_F = 1.0;  eps_A = 13.0+5.0i; eps_F = 1.0; 
phi_A1 = 0.0; phi_A2 = 0.0;           % phi_A1 = 0.1; phi_A2 = 1.1;      phi_A1 = 0.0; phi_A2 = 0.8;
delta_A = 2;                          % delta_A = 6; 
alpha_F = 0.0; beta_F = 0.0;          % alpha_F = 0.3; beta_F = 0.1;      alpha_F = 0.5; beta_F = 0.5;



% q: Bloch wavenumber
Nq = 120;
% q_vec = linspace(0.02,1-0.02,Nq);
q_vec = linspace(-0.5,0.5,Nq);

% k0: initial guess, can attain positive and negative imaginary part
k_max = 2.0;
Nk = 51;
k0_vec = [ linspace(0,k_max,Nk), linspace(0,k_max,Nk);
                    0.1*ones(1,Nk), -0.1*ones(1,Nk) ];

kk_full = NaN * zeros(Nk,Nq);
tol = 5e-4;

options = optimset('TolFun',1e-14,'TolX',1e-14);

fq_vec = zeros(2,Nk);

% solve all the roots for each fixed q
% ----------------------------------------------------------------------
for nq = 1:1:Nq
    
    q = q_vec(nq);

    % initial guess of roots
    for j = 1:Nk
        fq_vec(:,j) = fun_dispersion_three_layers(k0_vec(:,j),q);
    end
    fq = fq_vec(1,:) + 1i*fq_vec(2,:);
    
    k0_vec_q = k0_vec(:,find(abs(fq) < 1e4));
    Nk_q = size(k0_vec_q,2);
    
    k_vec = 100 * ones(1,Nk_q);
    
    for nk = 1:1:Nk_q
        k0 = k0_vec_q(:,nk);
        fun = @(k)fun_dispersion_three_layers(k,q);
        [kk, fval, flag] = fsolve(fun,k0,options);   % solve the algebraic equation (2)
        if ( flag > 0)
            k_vec(nk) = kk(1) + 1i*kk(2);
        end
    end
    
    % sort the roots in an ascending order of their real parts
    [~,Idx] = sort(real(k_vec));             
    k_vec = k_vec(Idx);
    
    % sort the roots with the same real parts by their imaginar parts in an
    % ascending order
    idx_real_diff = [0, find(abs(diff(real(k_vec)))>1e-3), length(k_vec) ];
    for j = 1:length(idx_real_diff)-1
        start_j = idx_real_diff(j)+1; end_j = idx_real_diff(j+1);
        m = start_j:end_j;
        [~,Idx_j] = sort(imag(k_vec(m)));
        k_vec(m) = k_vec(idx_real_diff(j)+Idx_j);
    end
    
    k_vec = k_vec( (real(k_vec)>-1e-15) & (real(k_vec)<=k_max) );  % keep only the nongegative roots in the interval of interest
    
    kk_vec = [k_vec(1), k_vec(1+find(abs(diff(k_vec))>tol)) ];     % eleminate the redundancy (e.g., two roots too close)
    kk_full(1:length(kk_vec),nq) = kk_vec;
    
end
% ---------------------------------------------------------------------

% plot the spectral band
figure;
for n=1:6
    plot(q_vec,real(kk_full(n,:)));
    hold on; 
end
axis equal; axis([-0.5 0.5 0 k_max]);
% axis equal; axis([0 1 0 1]);

xlabel('$q$','Interpreter','latex'); ylabel('$\Re(\omega)$','Interpreter','latex');

figure;
for n=1:6
    plot(q_vec,imag(kk_full(n,:)));
    hold on;
end

axis equal; axis([-0.5 0.5 -0.1 0.1]);
xlabel('$q$','Interpreter','latex'); ylabel('$\Im(\omega)$','Interpreter','latex');


figure;
for n=1:6
    hold on;
    plot(real(kk_full(n,:)), imag(kk_full(n,:))); % ,'r-o' );
end
