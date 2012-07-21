<?php

	require_once(dirname(__FILE__).'/Exceptions.php');
	require_once(dirname(__FILE__).'/MatrixOps.php');
	require_once(dirname(__FILE__).'/Statistics.php');
	require_once(dirname(__FILE__).'/GradientDescent.php');
	require_once(dirname(__FILE__).'/LinearRegression.php');

	function testOn($data, $test, $sq = FALSE)
	{
		$lr = LinearRegression::make();
	
		$lr->setLog(function($x) { print($x."\n"); });
	
		$lr->setData($data);
	
		$lr->train();
	
		$f = $lr->mkFunc();
	
		foreach ($test as $x)
		{
			if ($sq)
			{
				print($x.' '.$f($x, $x * $x)."\n");
			}
			else
			{
				print($x.' '.$f($x)."\n");
			}
		}
	}

	testOn(array(
			array(0, 70),
			array(60, 73),
			array(120, 72),
			array(179, 82),
			array(240, 84),
			),
			array(0, 60, 120, 179, 240, 315));

	testOn(array(
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
			array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891));

	testOn(array(
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
			), array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10));

	testOn(array(
			array(-2, -2 * -2, 7.997),
			array(-1, -1 * -1, 2.995),
			array(0, 0 * 0, 0.002),
			array(1, 1 * 1, -0.995),
			array(2, 2 * 2, -0.0001),
			array(3, 3 * 3, 3.0025),
			array(4, 4 * 4, 8.0005),
			), array(-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6), TRUE);

	testOn(array(
			array(0, 0, 70),
			array(60, 60 * 60, 73),
			array(120, 120 * 120, 72),
			array(179, 179 * 179, 82),
			array(240, 240 * 240, 84),
			), array(0, 60, 120, 179, 240, 315), TRUE);

	testOn(array(
			array(10, 10 * 10, 94),
			array(71, 71 * 71, 94),
			array(129, 129 * 129, 87),
			array(189, 189 * 189, 84),
			array(250, 250 * 250, 78),
			array(310, 310 * 310, 79),
			array(370, 370 * 370, 86),
			array(430, 430 * 430, 78),
			array(490, 490 * 490, 76),
			array(550, 550 * 550, 80),
			), array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891), TRUE);

?>
