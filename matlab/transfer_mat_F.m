function T = transfer_mat_F(k,L,eps,alpha,beta)

% q: momentum
% k: frequency
% L: thickness of the layer
% alpha, beta: perturbation parameters


% eps_F = [ eps   i*a;
%          -i*a   eps  ]
% mu_F = [ mu      i*beta;
%          -i*beta   mu   ]


% eps = 1; 
mu = 1;
a = k*L;	

n1 = sqrt((eps+alpha)*(mu+beta));
n2 = sqrt((eps-alpha)*(mu-beta));
m1 = sqrt((eps+alpha)/(mu+beta));
m2 = sqrt((eps-alpha)/(mu-beta));

u1 = cos(n1*a); v1 = sin(n1*a);
u2 = cos(n2*a); v2 = sin(n2*a);

T = [   u1+u2              1i*(u1-u2)           v1/m1-v2/m2      1i*v1/m1+1i*v2/m2;
      -1i*(u1-u2)            u1+u2          -1i*(v1/m1+v2/m2)      v1/m1-v2/m2;
      -m1*v1+m2*v2        -1i*(m1*v1+m2*v2)         u1+u2            1i*(u1-u2);
      1i*(m1*v1+m2*v2)    -m1*v1+m2*v2           -1i*(u1-u2)         u1+u2        ];

T = 1/2 * T;

end
