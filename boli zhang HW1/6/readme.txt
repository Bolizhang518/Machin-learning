Logistic regression cost function:
 	J(theta) = (1/m) Sum from i = 1 to m of costFunction(h_theta(x^(i)), y^(i)) 
	costFunction(h_theta(x), y)) = { -log(h_theta(x)) if y = 1; -log(1 - h_theta(x)) if y = 0
	y always equal 0 
	costFunction(h_theta(x), y)) = -y * log(h_theta(x)) - (1 - y) * log(1 - h_theta(x))
	principle of maximum likelyness estimation :
	J(theta) = (-1/m) Sum from i = 1 to m of [y^(i) * log(h_theta(x^(i))) + (1 - y^(i)) * log(1 - h_theta(x^(i)))]
	smallest: 
	h_theta(x) = 1/(1 + e^(-theta^T * x)) 
	
#1 compute J
#2 compute j(theta)  (for j=0 ,1,2..n)
#3 plug the result #2 to gradient descent 



	