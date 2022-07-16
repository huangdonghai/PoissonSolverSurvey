%plots a plane where plane(1) = a, plane(2) = b, plane(3) = c, for the
%generic equation ax + by +cz = 1

function h = plot_plane(pointA,pointB,pointC)
%{
hold on
a = plane(1);
b = plane(2);
c = plane(3);

%finding a point on the plan
A = [a b c; a b c; a b c];
point = pinv(A)*[1;1;1];
point = point';

%defining a vector normal to the plane
normal = [a,b,c];

%the plane is ax + by + cz  = 1
d = -point*normal';

% creates x,y
[xx,yy]=ndgrid(-1.8:.01:1.6,-1.8:.01:1.6);

% calculate corresponding zs
z = (-normal(1)*xx - normal(2)*yy - d)/normal(3);

%# plot the surface
color = ones(size(z));
surf(xx,yy,z, color')

hold off
%}

hold on
normal = cross(pointA-pointB, pointA-pointC); %# Calculate plane normal
%# Transform points to x,y,z
x = [pointA(1) pointB(1) pointC(1)];  
y = [pointA(2) pointB(2) pointC(2)];
z = [pointA(3) pointB(3) pointC(3)];

%Find all coefficients of plane equation    
A = normal(1); B = normal(2); C = normal(3);
D = -dot(normal,pointA);
%Decide on a suitable showing range
xLim = [-4 4];
zLim = [-50 250];
[X,Z] = meshgrid(xLim,zLim);
Y = (A * X + C * Z + D)/ (-B);
reOrder = [1 2  4 3];
h = patch(X(reOrder),Y(reOrder),Z(reOrder),'r');
alpha(h,.3)
%alpha(0.3);


end