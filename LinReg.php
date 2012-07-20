<?php

	class MlException extends Exception
	{
	}

	final class LinReg
	{
		private $data;
		private $x;
		private $y;
		private $m;
		private $dim;
		private $theta = array(); // current hypothesis
		private $alpha; // gradient descent step size
		private $convThreshold; // cost function threshold at which we have converged to optimum
		private $maxIters; // max. number of gradient descent iterations
		private $log = NULL;

		private $xm;
		private $xs;
		private $ym;
		private $ys;

		final static public function make()
		{
			return new self;
		}

		final public function setData(array $data)
		{
			$this->data = $data;
			$this->m = count($this->data);
			$this->dim = count($this->data[0]);

			if ($this->dim !== count($this->theta))
			{
				$this->setTheta(array_fill(0, $this->dim, 0));
			}

			$this->x = array();
			$this->y = array();

			foreach ($this->data as $obs)
			{
				$x = $obs;
				$y = array_pop($x);
				$this->x[] = $x;
				$this->y[] = $y;
			}

			$this->x = M::transpose($this->x);

			$nx = array_map(function($x) { return Stat::norm($x); }, $this->x);

			$this->x = array(array_fill(0, $this->m, 1));
			$this->xm = array(0);
			$this->xs = array(1);

			foreach ($nx as $xl)
			{
				list($xm, $xs, $x) = $xl;
				$this->xm[] = $xm;
				$this->xs[] = $xs;
				$this->x[] = $x;
			}

			$this->x = M::transpose($this->x);

			list($this->ym, $this->ys, $this->y) = Stat::norm($this->y);

			return $this;
		}

		final public function setTheta(array $theta)
		{
			if (count($theta) !== $this->dim)
			{
				throw new MlException('Invalid dimensionality of theta: expected '.$this->dim.' elements');
			}

			$this->theta = $theta;

			return $this;
		}

		final public function setAlpha($alpha)
		{
			$this->alpha = floatval($alpha);
			
			return $this;
		}

		final public function setMaxIters($maxIters)
		{
			$this->maxIters = intval($maxIters);

			return $this;
		}

		final public function setConvThreshold($convThreshold)
		{
			$this->convThreshold = floatval($convThreshold);

			return $this;
		}

		final public function setLog($log)
		{
			$this->log = $log;

			return $this;
		}

		final public function train()
		{
			$self = $this;

			$gd = GradDesc::make(function(array $t) use($self) { return $self->costGradFunc($t); }, $this->theta,
					$this->maxIters, $this->convThreshold, $this->alpha, TRUE);

			$gd->setLog($this->log);

			$theta = $gd->descend();

			$this->theta = $theta;
		}

		final public function costGradFunc(array $theta)
		{
			$cost = 0;
			$grad = array_fill(0, $this->dim, 0);

			$m = $this->m;

			for ($i = 0; $i != $this->m; ++$i)
			{
				$h = $this->estimate($theta, $this->x[$i]);

				$d = $h - $this->y[$i];

				$cost += ($d * $d);

				$grad = array_map(function($g, $x) use($d) { return $g + ($d * $x); }, $grad, $this->x[$i]);
			}

			return array($cost / (2 * $m), array_map(function($x) use($m) { return $x / $m; }, $grad));
		}

		final public function estimate(array $theta, array $x)
		{
			return array_sum(array_map(function($t, $x) { return $t * $x; }, $theta, $x));
		}

		final public function mkFunc()
		{
			$self = $this;

			$theta = $this->theta;
			$xm = $this->xm;
			$xs = $this->xs;
			$ym = $this->ym;
			$ys = $this->ys;

			return
					function() use($self, $theta, $xm, $xs, $ym, $ys)
					{
						$x = func_get_args();

						array_unshift($x, 1);

						$x = array_map(function($x, $m, $s) { return ($x - $m) / $s; }, $x, $xm, $xs);

						return ($self->estimate($theta, $x) * $ys) + $ym;
					};
		}

		final private function __construct()
		{
			$this->setData(array(array(0, 0), array(1, 1)));
			$this->setAlpha(0.1);
			$this->setConvThreshold(1e-6);
			$this->setMaxIters(500);
		}
	}

	final class GradDesc
	{
		private $f;
		private $theta;
		private $iter;
		private $maxIters;
		private $convThreshold;
		private $alpha;
		private $flexibleAlpha;
		private $log = NULL;

		final static public function make($f, array $theta,
				$maxIters, $convThreshold, $alpha, $flexibleAlpha = TRUE)
		{
			return new self($f, $theta,
					$maxIters, $convThreshold, $alpha, $flexibleAlpha);
		}

		final public function setLog($log)
		{
			$this->log = $log;

			return $this;
		}

		final public function descend()
		{
			$f = $this->f;

			list($curCost, $grad) = $f($this->theta);
			$thr = $this->convThreshold;

			$alpha = $this->alpha;

			while (($this->iter <= $this->maxIters) && ($thr >= $this->convThreshold))
			{
				$this->log('Iter #'.$this->iter.' Cost: '.$curCost.' Thr: '.$thr.' Alpha: '.$alpha);
				$this->log('Theta '.implode(', ', $this->theta));

				$newTheta = array_map(function ($t, $g) use($alpha) { return $t - ($alpha * $g); }, $this->theta, $grad);

				$this->log('Theta candidate '.implode(', ', $newTheta));

				list($newCost, $grad) = $f($newTheta);

				if ($newCost > $curCost)
				{
					$alpha /= 2;

					$this->log('Cost grew to '.$newCost.', reducing alpha to '.$alpha);

					if ($alpha === 0)
					{
						throw new MlExcpetion('Alpha reached zero');
					}

					continue;
				}

				$thr = $curCost - $newCost;
				$curCost = $newCost;
				$this->theta = $newTheta;

				++$this->iter;
			}

			$this->log('Theta '.implode(', ', $this->theta));

			return $this->theta;
		}

		final private function __construct($f, array $theta,
				$maxIters, $convThreshold, $alpha, $flexibleAlpha = TRUE)
		{
			if (! is_callable($f))
			{
				throw new MlException('Function for computing cost and gradient is required');
			}

			$this->f = $f;
			$this->theta = $theta;
			$this->iter = 1;
			$this->maxIters = intval($maxIters);
			$this->convThreshold = floatval($convThreshold);
			$this->alpha = floatval($alpha);
			$this->flexibleAlpha = $flexibleAlpha;
		}

		final private function log($str)
		{
			if (is_callable($this->log))
			{
				$log = $this->log;

				$log($str);
			}
		}
	}

	final class Stat
	{
		final static public function mean(array $x)
		{
			return array_sum($x) / count($x);
		}

		final static public function vari(array $x)
		{
			$m = self::mean($x);

			$z = array_map(function($x) use($m) { return ($x - $m) * ($x - $m); }, $x);

			return self::mean($z);
		}

		final static public function stddev(array $x)
		{
			return sqrt(self::vari($x));
		}

		final static public function norm(array $x)
		{
			$m = self::mean($x);
			$s = self::stddev($x);

			return array($m, $s, array_map(function($x) use($m, $s) { return ($x - $m) / $s; }, $x));
		}
	}

	final class M
	{
		final static public function transpose(array $x)
		{
			$res = array_fill(0, count($x[0]), array());

			foreach ($x as $line)
			{
				foreach ($line as $i => $elt)
				{
					$res[$i][] = $elt;
				}
			}

			return $res;
		}
	}

	$lr = LinReg::make();

	$lr->setLog(function($x) { print($x."\n"); });

	$lr->setData(array(
array(0, 70),
array(60, 73),
array(120, 72),
array(179, 82),
array(240, 84),
			));

	$lr->train();

	$f = $lr->mkFunc();

	foreach (array(0, 60, 120, 179, 240, 315) as $x)
	{
		print($x.' '.$f($x)."\n");
	}

	$lr = LinReg::make();

	$lr->setLog(function($x) { print($x."\n"); });

	$lr->setData(array(
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
			));

	$lr->train();

	$f = $lr->mkFunc();

	foreach (array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891) as $x)
	{
		print($x.' '.$f($x)."\n");
	}

	$lr = LinReg::make();

	$lr->setLog(function($x) { print($x."\n"); });

	$lr->setData(array(
array(0, 0, 70),
array(60, 60 * 60, 73),
array(120, 120 * 120, 72),
array(179, 179 * 179, 82),
array(240, 240 * 240, 84),
			));

	$lr->train();

	$f = $lr->mkFunc();

	foreach (array(0, 60, 120, 179, 240, 315) as $x)
	{
		print($x.' '.$f($x, $x * $x)."\n");
	}

	$lr = LinReg::make();

	$lr->setLog(function($x) { print($x."\n"); });

	$lr->setData(array(
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
			));

	$lr->train();

	$f = $lr->mkFunc();

	foreach (array(10, 71, 129, 189, 250, 310, 370, 430, 490, 550, 891) as $x)
	{
		print($x.' '.$f($x, $x * $x)."\n");
	}

?>
