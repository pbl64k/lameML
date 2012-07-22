<?php

	abstract class FooRegression
	{
		protected $data;
		protected $x;
		protected $y;
		protected $m;
		protected $dim;
		protected $enrichment;
		/**
		 * Current hypothesis - list of parameters of a linear approximation
		 */
		protected $theta = array();
		/**
		 * Regularization coefficient, set to zero to run unregularized regression
		 */
		protected $lambda;
		/**
		 * Gradient descent step size
		 */
		protected $alpha;
		/**
		 * Cost function change threshold at which we have converged
		 * close enough to optimum
		 */
		protected $convThreshold;
		/**
		 * Max. number of gradient descent iterations
		 */
		protected $maxIters;
		protected $normalizeY;
		protected $log = NULL;

		/**
		 * Means and standard deviations of the original data
		 * Needed for normalization and to recover unnormalized values
		 */
		protected $xm;
		protected $xs;
		protected $ym;
		protected $ys;

		final public function setData(array $data, $enrichment = 1)
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

			$this->enrichment = $enrichment;

			$this->preprocessData();

			$this->enrichData($enrichment);

			$this->dim = count($this->x[0]) + 1;

			if ($this->dim !== count($this->theta))
			{
				$this->resetTheta();
			}

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

		final public function setLambda($lambda)
		{
			$this->lambda = floatval($lambda);
			
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

		final public function setNormalizeY($normY)
		{
			$this->normalizeY = $normY;

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

			return $this;
		}

		abstract public function costGradFunc(array $theta);

		abstract public function estimate(array $theta, array $x);

		abstract public function mkFunc();

		final public function enrichRow($row, $enrichment)
		{
			if ($enrichment == 1)
			{
				return $row;
			}

			$result = $row;

			foreach ($row as $i => $x)
			{
				$xe = $this->enrichRow(array_slice($row, $i), $enrichment - 1);

				$result = array_merge($result, array_map(
						function($xe) use($x)
						{
							return $xe * $x;
						}, $xe));
			}

			return $result;
		}

		protected function __construct()
		{
			$this->setData(array(array(0, 0), array(1, 1)));
			$this->setLambda(0);
			$this->setAlpha(1);
			$this->setConvThreshold(1e-6);
			$this->setMaxIters(5000);
			$this->setNormalizeY(FALSE);
		}

		final protected function preprocessData()
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

			return $this;
		}

		final protected function enrichData($enrichment)
		{
			foreach ($this->x as $i => $x)
			{
				$this->x[$i] = $this->enrichRow($x, $enrichment);
			}

			return $this;
		}

		final protected function normalizeData()
		{
			$this->x = MatrixOps::transpose($this->x);

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

			if ($this->normalizeY)
			{
				list($this->ym, $this->ys, $this->y) = Statistics::normalize($this->y);
			}

			return $this;
		}
	}


?>
