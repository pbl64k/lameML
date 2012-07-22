<?php

	/**
	 * Approximates a given data set by a linear function
	 */
	final class LinearRegression extends FooRegression
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
				$h = $this->estimate($theta, $this->x[$i]);

				$d = $h - $this->y[$i];

				$cost += ($d * $d);

				$grad = array_map(
						function($grad, $x) use($d)
						{
							return $grad + ($d * $x);
						}, $grad, $this->x[$i]);
			}

			$regTheta = $theta;

			// We do not regularize the intercept term
			$regTheta[0] = 0;

			return array(
					($cost / (2 * $m)) +
					($lambda * array_sum(array_map(
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

			$enrichment = $this->enrichment;

			$enrich =
					function(array $x) use($self, $enrichment)
					{
						return $self->enrichRow($x, $enrichment);
					};

			return
					function() use($self, $theta, $xm, $xs, $ym, $ys, $enrich)
					{
						$x = func_get_args();

						$x = $enrich($x);

						array_unshift($x, 1);

						$x = array_map(
								function($x, $m, $s)
								{
									return ($x - $m) / $s;
								}, $x, $xm, $xs);

						return ($self->estimate($theta, $x) * $ys) + $ym;
					};
		}

		final protected function __construct()
		{
			parent::__construct();

			$this->setNormalizeY(TRUE);
		}
	}

?>
