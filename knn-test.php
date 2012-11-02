<?php

	$c = array();

	$h = fopen(dirname(__FILE__).'/train.csv', 'r');

	$hd = fgetcsv($h);

	$data = array();

	while(($r = fgetcsv($h)) !== FALSE)
	{
		$tr = array(
				// read features
				);

		$rr = $tr;

		$data[] = $rr;
		$c[] = $r[0] == 1 ? 1 : -1;
	}

	$n = count($data);

	$h = fopen(dirname(__FILE__).'/test.csv', 'r');

	$hd = fgetcsv($h);

	$test = array();

	while(($r = fgetcsv($h)) !== FALSE)
	{
		$tr = array(
				// read features
				);

		$test[] = $tr;
	}

	// Transform to z-scores

	$dim = count($data[0]);

	$means = array_fill(0, $dim, 0);

	foreach ($data as $d)
	{
		$means = array_map(
				function($a, $b)
				{
					return $a + $b;
				}, $means, $d);
	}

	$means = array_map(
			function($x) use($n)
			{
				return $x / $n;
			}, $means);

	$var = array_fill(0, $dim, 0);

	foreach ($data as $d)
	{
		$var = array_map(
				function($a, $b, $c)
				{
					return $a + (($c - $b) * ($c - $b));
				}, $var, $d, $means);
	}

	$var = array_map(
			function($x) use($n)
			{
				return $x / $n;
			}, $var);

	$sd = array_map(
			function($x)
			{
				return sqrt($x);
			}, $var);

	$remidx = array();

	foreach ($sd as $ix => $sd0)
	{
		if (abs($sd0) <= 1e-16)
		{
			$remidx[] = $ix;
		}
	}

	$f =
			function($x) use($remidx)
			{
				$res = array();

				foreach ($x as $ix => $x0)
				{
					if (in_array($ix, $remidx))
					{
						continue;
					}

					$res[] = $x0;
				}

				return $res;
			};

	$means = $f($means);
	$var = $f($var);
	$sd = $f($sd);

	$z = function($d) use($means, $sd)
			{
				$res = array();

				foreach ($d as $ix => $x)
				{
					$res[] = ($x - $means[$ix]) / $sd[$ix];
				}

				return $res;
			};

	$data = array_map($f, $data);
	$data = array_map($z, $data);

	$dim = count($data[0]);

	$kdt = KdTree::make($dim);

	foreach ($data as $ix => $d)
	{
		$kdt->add($d, $c[$ix]);
	}

	foreach ($test as $ix => $d)
	{
		$d = $z($f($d));

		$nn = $kdt->findKnn($d, 10);

		$cat = array_sum(array_map(
				function($n)
				{
					// alternatively, count vote without weighing by inverse of distance
					return ((abs($n[0]) < 1e-8) ? 1e8 : (1 / $n[0])) * $n[2];
				}, $nn));

		print((($cat >= 0) ? '1' : '0')."\n");
	}

?>
