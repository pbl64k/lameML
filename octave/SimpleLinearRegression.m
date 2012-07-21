
function [a, b] = SimpleLinearRegression(x, y);

	b = cov(x, y) / var(x);
	a = mean(y) - (b * mean(x));

end;

