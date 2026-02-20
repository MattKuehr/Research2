function T = transfer_mat_A(k,L,eps,delta,phi)

% q: momentum
% k: frequency
% L: thickness of the layer
% delta: perturbation
% phi: rotation angle


% eps_A = [ eps+delta*cos(2*phi) delta*sin(2*phi);
%           delta*sin(2*phi)     eps-delta*cos(2*phi) ]


a = k*L;

n1 = sqrt(eps + delta);
n2 = sqrt(eps - delta);

u1 = cos(n1*a); v1 = sin(n1*a);
u2 = cos(n2*a); v2 = sin(n2*a);

u = cos(phi); v = sin(phi);

T = [     u*u*u1 + v^2*u2             u*v*u1 - u*v*u2              -1i*u*v*v1/n1 + 1i*u*v*v2/n2     1i*u*u*v1/n1 + 1i*v*v*v2/n2;
          u*v*u1 - u*v*u2             v*v*u1 + u*u*u2              -1i*v*v*v1/n1 - 1i*u*u*v2/n2     1i*u*v*v1/n1 - 1i*u*v*v2/n2;
      -1i*n1*u*v*v1 + 1i*n2*u*v*v2  -1i*n1*v*v*v1 - 1i*n2*u*u*v2        v*v*u1 + u*u*u2               -u*v*u1+u*v*u2;
       1i*n1*u*u*v1 + 1i*n2*v*v*v2   1i*n1*u*v*v1 - 1i*n2*u*v*v2       -u*v*u1 + u*v*u2                u*u*u1+v*v*u2            ];


end

