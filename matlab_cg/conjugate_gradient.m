warning ('off','all');

clc
clear

% define out Ax = b problem, linear system of equations
A = [5 2; 2 3]
b = [-2; 4]

%The quadractic form is 
f = @(x1,x2) 0.5.*[x1;x2]'*A*[x1; x2] -b'*[x1; x2];
ezsurf(f)%plots function

%define gradient as Ax-b
f_grad = @(x1,x2) A*[x1;x2] - b;
% this is the case when A is symmetric and postitive definite

%% running conjugate gradient
counter = 0;
x1 = 5; x2 = 2; %inital guess
x = [x1;x2];
tol = 1e-4; %tolerance for convergence


r = -f_grad(x(1),x(2));
d=r;
rsold=r'*r;

for i=1:10^(6)
    
    %plotting 
    ezsurf(f)
    axis square
    hold on
    scatter3(x(1),x(2),f(x(1),x(2)),'black','fill') %current location of solution
    
    counter = counter + 1;
    alpha=rsold/(d'*A*d);
    p1 = [x(1),x(2),f(x(1),x(2))];
    x=x+alpha*d
    r=r-alpha*A*d;
    
    %more plotting
    p2 = [x(1),x(2),f(x(1),x(2))];
    scatter3(x(1),x(2),f(x(1),x(2)),'red','fill') %next location of solution
    plot_plane([x(1),x(2),1],[x(1),x(2),1]+[d(1),d(2),0],[x(1),x(2),10]);
    line([p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],'LineWidth',4,'Color',[.8 .8 .8]);%new location of solution
    pause
    
    rsnew=r'*r;
    if sqrt(rsnew)<1e-10
        break;
    end
    d=r+rsnew/rsold*d;
    rsold=rsnew;
    
end

x = [x(1);x(2)];

str = ['The point (' , num2str(x(1)) , ',', num2str(x(2)), ') was found to be a' ];
disp(str);
str = ['local minimum in ', num2str(counter), ' iterations' ];
disp(str);