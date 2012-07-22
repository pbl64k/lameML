<?php

	/**
	 * Attempts to find a function's minimum by naive gradient descent
	 */
	final class GradientDescent
	{
		/**
		 * Function to minimize
		 */
		private $f;
		/**
		 * Current position in the solution space
		 */
		private $theta;
		/**
		 * Number of current iteration
		 */
		private $iter;
		/**
		 * Maximum number of iterations - search will be terminated upon reaching this
		 */
		private $maxIters;
		/**
		 * Threshold of df - once a step's df goes below this, we consider the current
		 * position to be close enough to the optimum
		 */
		private $convThreshold;
		/**
		 * Gradient descent's step size - gradient multiplied by this before applying
		 * to current parameters
		 */
		private $alpha;
		/**
		 * Whether alpha should be reduced or not if we start diverging
		 * This is not necessarily a very good idea.
		 */
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

			list($fval, $grad) = $f($this->theta);

			$df = $this->convThreshold;

			$alpha = $this->alpha;

			while (($this->iter <= $this->maxIters) && ($df >= $this->convThreshold))
			{
				$this->log('Iteration #'.$this->iter);
				$this->log('f: '.$fval.' df: '.$df.' alpha: '.$alpha);
				$this->log('theta '.implode(', ', $this->theta));

				$newTheta = array_map(
						function ($theta, $grad) use($alpha)
						{
							return $theta - ($alpha * $grad);
						}, $this->theta, $grad);

				$this->log('theta candidate '.implode(', ', $newTheta));

				list($newFval, $grad) = $f($newTheta);

				if ($newFval > $fval)
				{
					$alpha /= 2;

					$this->log('f grew to '.$newFval.', reducing alpha to '.$alpha);

					if ($alpha === 0)
					{
						throw new MlException('alpha reached zero');
					}

					continue;
				}

				$df = $fval - $newFval;
				$fval = $newFval;
				$this->theta = $newTheta;

				++$this->iter;
			}

			$this->log('final theta '.implode(', ', $this->theta));

			return $this->theta;
		}

		final private function __construct($f, array $theta,
				$maxIters, $convThreshold, $alpha, $flexibleAlpha = TRUE)
		{
			if (! is_callable($f))
			{
				throw new MlException('Function computing value function and its gradient is required');
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

			return $this;
		}
	}

?>
