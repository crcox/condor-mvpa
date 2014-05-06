function y = roundto(x,multiple)
	if multiple < 2
		y = x;
	end

	m = mod(x,multiple);

	if (m/multiple)<0.5
		y = x-m;
	else
		y = x-m+multiple;
	end
end
