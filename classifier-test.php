<?php

	require_once(dirname(__FILE__).'/Exceptions.php');
	require_once(dirname(__FILE__).'/MatrixOps.php');
	require_once(dirname(__FILE__).'/Statistics.php');
	require_once(dirname(__FILE__).'/GradientDescent.php');
	require_once(dirname(__FILE__).'/FooRegression.php');
	require_once(dirname(__FILE__).'/LogisticRegression.php');

	function testOn(array $arg)
	{
		$data = $arg[0];
		$test = $arg[1];
		$lambda = $arg[2];
		$enrichment = $arg[3];

		$lr = LogisticRegression::make();
	
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
			array(0, 0, 1),
			array(-0.5, 1, 1),
			array(3, 0, 0),
			array(0.5, 5, 0),
			array(-0.75, 2, 0),
			array(-0.75, -0.5, 1),
			array(-1, 0, 1),
			array(0, -0.25, 1),
			array(-5, 0, 0),
			array(5, 2, 0),
			array(-2, -2, 0),
			),
			array(
			array(0, 0),
			array(-0.5, 1),
			array(3, 0),
			array(0.5, 5),
			array(-0.75, 2),
			array(-0.75, -0.5),
			array(-1, 0),
			array(0, -0.25),
			array(-5, 0),
			array(5, 2),
			array(-2, -2),
			array(0.5, 0.5),
			array(-1, -1),
			array(-0.5, 0),
			array(3, 3),
			array(-2, 3),
			array(5, -0.5),
			),
			0.0001,
			2);

	$tests[2] = array(array(
			array(0, 0, 0),
			array(0, 1, 0),
			array(0, 2, 0),
			array(0, 3, 0),
			array(0, 4, 0),
			array(3, 0, 1),
			array(3, 1, 1),
			array(3, 2, 1),
			array(3, 3, 1),
			array(3, 4, 1),
			),
			array(
			array(0, 0),
			array(0, 1),
			array(0, 2),
			array(0, 3),
			array(0, 4),
			array(3, 0),
			array(3, 1),
			array(3, 2),
			array(3, 3),
			array(3, 4),
			array(0, -1),
			array(0, 5),
			array(3, -1),
			array(3, 5),
			array(2, 2),
			array(1, 1),
			array(-1, 0),
			array(4, 0),
			),
			0,
			1);

	if ((! array_key_exists(1, $argv)) || (! array_key_exists(intval($argv[1]), $tests)))
	{
		$argv[1] = 1;
	}

	testOn($tests[$argv[1]]);

?>
