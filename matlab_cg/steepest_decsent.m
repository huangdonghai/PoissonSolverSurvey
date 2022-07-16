warning ('off','all');

clc
clear

% define out Ax = b problem, linear system of equations
A = [5 2; 2 3];
b = [-2; 4];

%The quadractic form is 
f = @(x1,x2) 0.5.*[x1;x2]'*A*[x1; x2] -b'*[x1; x2];
ezsurf(f)%plots function

%define gradient
f_grad = @(x1,x2) A*[x1;x2] - b;
% this is the case when A is symmetric and postitive definite

%% running steepest descent
x1 = 5; x2 = 2; %inital guess
tol = 1e-4; %tolerance for convergence
u = zeros(2,1); x_er = 1000; y_er = 1000; counter = 0;

while x_er > tol || y_er > tol
    direction_of_grad = f_grad(x1,x2);
    u(1) = direction_of_grad(1); %gradient in x1 direction
    u(2) = direction_of_grad(2); %gradient in x2 direction
    
    g = @(t) f(x1 - t*u(1),x2 - t*u(2)); %function for 1D minimization
    %T = fminsearch(g,0) %1D minimization to find t*
    r = -f_grad(x1,x2);
    T = (r'*r)/(r'*A*r);
    
    p1 = [x1,x2,f(x1,x2)];
    
    %stuff just for plotting purposes
    ezsurf(f)%plots function
    axis square
    hold on
    scatter3(x1,x2,f(x1,x2),'black','fill') %current location of solution
    
    %update x y guess
    x1 = x1 - T*u(1);
    x2 = x2 - T*u(2);
    
    %update convergence metrics
    errors = abs(f_grad(x1,x2)); %want gradient to be zero
    x_er = errors(1);
    y_er = errors(2);
    
    counter = counter + 1; %count interations
    
    %more plotting
    h = plot_plane([x1,x2,1],[x1,x2,1]+[u(1),u(2),0],[x1,x2,10]);
    scatter3(x1,x2,f(x1,x2),'red','fill') %next location of solution
    p2 = [x1,x2,f(x1,x2)];
    line([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',4,'Color',[.8 .8 .8]);%new location of solution
    pause
    delete(h)  
end

x = [x1;x2];

str = ['The point (' , num2str(x1) , ',', num2str(x2), ') was found to be a' ];
disp(str);
str = ['local minimum in ', num2str(counter), ' iterations' ];
disp(str);