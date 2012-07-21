
[a, b] = SimpleLinearRegression([0; 60; 120; 179; 240], [70; 73; 72; 82; 84]);

x = [0, 60, 120, 179, 240, 315];

for i = 1:length(x);
	disp(a + b * x(i));
end;

