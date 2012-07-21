<?php

	/**
	 * Approximates a given data set by a linear function
	 */
	final class LinearRegression
	{
		private $data;
		private $x;
		private $y;
		private $m;
		private $dim;
		/**
		 * Current hypothesis - list of parameters of a linear approximation
		 */
		private $theta = array();
		/**
		 * Gradient descent step size
		 */
		private $alpha;
		/**
		 * Cost function change threshold at which we have converged
		 * close enough to optimum
		 */
		private $convThreshold;
		/**
		 * Max. number of gradient descent iterations
		 */
		private $maxIters;
		private $log = NULL;

		/**
		 * Means and standard deviations of the original data
		 * Needed for normalization and to recover unnormalized values
		 */
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

			if ($this->m < 2)
			{
				throw new MlException('Empty or degenerate data set');
			}

			$this->dim = count($this->data[0]);

			if ($this->dim < 2)
			{
				throw new MlException('Dimensionality of input should be at least 2');
			}

			if ($this->dim !== count($this->theta))
			{
				$this->resetTheta();
			}

			$this->preprocessData();

			$this->normalizeData();

			return $this;
		}

		final public function setTheta(array $theta)
		{
			if (count($theta) !== $this->dim)
			{
				throw new MlException(
						'Invalid dimensionality of theta: expected '.$this->dim.' elements');
			}

			$this->theta = $theta;

			return $this;
		}

		final public function resetTheta()
		{
			$this->setTheta(array_fill(0, $this->dim, 0));

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

			$gd = GradientDescent::make(
					function(array $theta) use($self)
					{
						return $self->costGradFunc($theta);
					},
					$this->theta, $this->maxIters, $this->convThreshold, $this->alpha, TRUE);

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

				$grad = array_map(
						function($grad, $x) use($d)
						{
							return $grad + ($d * $x);
						}, $grad, $this->x[$i]);
			}

			return array(
					$cost / (2 * $m),
					array_map(
							function($x) use($m)
							{
								return $x / $m;
							}, $grad),
					);
		}

		final public function estimate(array $theta, array $x)
		{
			return array_sum(array_map(
					function($theta, $x)
					{
						return $theta * $x;
					}, $theta, $x));
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

						$x = array_map(
								function($x, $m, $s)
								{
									return ($x - $m) / $s;
								}, $x, $xm, $xs);

						return ($self->estimate($theta, $x) * $ys) + $ym;
					};
		}

		final private function __construct()
		{
			$this->setData(array(array(0, 0), array(1, 1)));
			$this->setAlpha(1);
			$this->setConvThreshold(1e-6);
			$this->setMaxIters(5000);
		}

		final private function preprocessData()
		{
			$this->x = array();
			$this->y = array();

			foreach ($this->data as $obs)
			{
				$x = $obs;
				$y = array_pop($x);
				$this->x[] = $x;
				$this->y[] = $y;
			}

			$this->x = MatrixOps::transpose($this->x);
		}

		final private function normalizeData()
		{
			$nx = array_map(
					function($x)
					{
						return Statistics::normalize($x);
					}, $this->x);

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

			$this->x = MatrixOps::transpose($this->x);

			list($this->ym, $this->ys, $this->y) = Statistics::normalize($this->y);
		}
	}

?>
