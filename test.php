<?php

	require_once(dirname(__FILE__).'/Exceptions.php');
	require_once(dirname(__FILE__).'/MatrixOps.php');
	require_once(dirname(__FILE__).'/Statistics.php');
	require_once(dirname(__FILE__).'/GradientDescent.php');
	require_once(dirname(__FILE__).'/FooRegression.php');
	require_once(dirname(__FILE__).'/LinearRegression.php');

	function testOn(array $arg)
	{
		$data = $arg[0];
		$test = $arg[1];
		$lambda = $arg[2];
		$enrichment = $arg[3];

		$lr = LinearRegression::make();
	
		$lr->setLog(function($x) { print($x."\n"); });
	
		$lr->setData($data, $enrichment);
		$lr->setLambda($lambda);
	
		$lr->train();
	
		$f = $lr->mkFunc();
	
		foreach ($test as $x)
		{
			if (is_array($x))
			{
				print('('.implode(', ', $x).') '.call_user_func_array($f, $x)."\n");
			}
			else
			{
				print($x.' '.$f($x)."\n");
			}
		}
	}

	$tests = array();

	$tests[1] = array(array(
			array(0, 70),
			array(60, 73),
			array(120, 72),
			array(179, 82),
			array(240, 84),
			),
			array(0, 60, 120, 179, 240, 315),
			0,
			1);

	$tests[2] = array(array(
			array(10, 94),
			array(71, 94),
			array(129, 87),
			array(189, 84),
			array(250, 78),
			array(310, 79),
			array(370, 86),
			array(430, 78),
			array(490, 76),
			array(550, 80),
			),
			array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891),
			0.01,
			1);

	$tests[3] = array(array(
			array(1, 15.001),
			array(2, 15.02),
			array(3, 14.9998),
			array(4, 15.003),
			array(5, 15.00001),
			array(6, 14.999),
			array(7, 14.999997),
			array(8, 15.002),
			array(9, 15.0003),
			array(10, 15.0005),
			), array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			0.01,
			1);

	$tests[4] = array(array(
			array(-2, 7.997),
			array(-1, 2.995),
			array(0, 0.002),
			array(1, -0.995),
			array(2, -0.0001),
			array(3, 3.0025),
			array(4, 8.0005),
			), array(-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
			0.000001,
			2);

	$tests[5] = array(array(
			array(0, 70),
			array(60, 73),
			array(120, 72),
			array(179, 82),
			array(240, 84),
			), array(0, 60, 120, 179, 240, 315),
			0.0001,
			2);

	$tests[6] = array(array(
			array(10, 94),
			array(71, 94),
			array(129, 87),
			array(189, 84),
			array(250, 78),
			array(310, 79),
			array(370, 86),
			array(430, 78),
			array(490, 76),
			array(550, 80),
			), array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891),
			0.0001,
			2);

	// x = a^3 + ab + 1
	$tests[7] = array(array(
			array(0, 0, 1),
			array(0, 1, 1),
			array(0, 2, 1),
			array(1, 0, 2),
			array(1, 1, 3),
			array(1, 2, 4),
			array(2, 0, 9),
			array(2, 1, 11),
			array(2, 2, 13),
			),
			array(
			array(0, 0),
			array(0, 1),
			array(0, 2),
			array(1, 0),
			array(1, 1),
			array(1, 2),
			array(2, 0),
			array(2, 1),
			array(2, 2),
			array(3, 3),
			),
			0,
			3);

	if ((! array_key_exists(1, $argv)) || (! array_key_exists(intval($argv[1]), $tests)))
	{
		$argv[1] = 1;
	}

	testOn($tests[$argv[1]]);

?>
