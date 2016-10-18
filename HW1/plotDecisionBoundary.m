function plotDecisionBoundary(theta, X, y  )

plotData1(X(:,2:3), y);
hold on
if size(X, 2) <= 3
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

% Calculate the decision boundary line

 plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

% Plot, and adjust axes for better viewing
plot(plot_x, plot_y)

 % Legend, specific for the exercise
 legend('low', 'Not low', 'Decision Boundary')
 
else
      u = linspace(-4000, 1.5, 50);
      v = linspace(-4000, 1.5, 50);
    
  z = zeros(length(u), length(v));
  % Evaluate z = theta*x over the grid
  for i = 1:length(u)
    for j = 1:length(v)
         z(i,j) = mapFeature(u(i), v(j))*theta;
    end
  end
  % important to transpose z before calling contour
  z = z';
  contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end

