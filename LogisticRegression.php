<?php

	final class LogisticRegression extends FooRegression
	{
		final static public function make()
		{
			return new self;
		}

		final public function costGradFunc(array $theta)
		{
			$cost = 0;
			$grad = array_fill(0, $this->dim, 0);

			$m = $this->m;
			$lambda = $this->lambda;

			for ($i = 0; $i != $this->m; ++$i)
			{
				$y = $this->y[$i];

				$h = $this->estimate($theta, $this->x[$i]);

				$d = ($y * log($h)) + ((1 - $y) * log(1 - $h));

				$cost += $d;

				$grad = array_map(
						function($grad, $x) use($h, $y)
						{
							return $grad + (($h - $y) * $x);
						}, $grad, $this->x[$i]);
			}

			$regTheta = $theta;

			// We do not regularize the intercept term
			$regTheta[0] = 0;

			return array(
					($cost / (-$m)) +
					(($lambda / (2 * $m)) * array_sum(array_map(
							function($theta)
							{
								return $theta * $theta;
							}, $regTheta))),
					array_map(
							function($x, $theta) use($m, $lambda)
							{
								return ($x + ($lambda * $theta)) / $m;
							}, $grad, $regTheta),
					);
		}

		final public function estimate(array $theta, array $x)
		{
			return $this->sigmoid(array_sum(array_map(
					function($theta, $x)
					{
						return $theta * $x;
					}, $theta, $x)));
		}

		final public function mkFunc()
		{
			$self = $this;

			$theta = $this->theta;
			$xm = $this->xm;
			$xs = $this->xs;

			$enrichment = $this->enrichment;

			$enrich =
					function(array $x) use($self, $enrichment)
					{
						return $self->enrichRow($x, $enrichment);
					};

			return
					function() use($self, $theta, $xm, $xs, $enrich)
					{
						$x = func_get_args();

						$x = $enrich($x);

						array_unshift($x, 1);

						$x = array_map(
								function($x, $m, $s)
								{
									return ($x - $m) / $s;
								}, $x, $xm, $xs);

						return $self->estimate($theta, $x);
					};
		}

		final public function sigmoid($x)
		{
			return 1 / (1 + exp(-$x));
		}

		final protected function __construct()
		{
			parent::__construct();
			$this->setNormalizeY(FALSE);
		}
	}

?>
