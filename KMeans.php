<?php

	/**
	 * Attempt to divide a population into clusters
	 */
	final class KMeans
	{
		private $data;
		private $c;
		private $cdata;
		private $means;
		private $iter;
		private $log = NULL;

		final static public function make()
		{
			return new self;
		}

		final public function setData($data)
		{
			$this->data = $data;
		}

		final public function setLog($log)
		{
			$this->log = $log;

			return $this;
		}

		final public function cluster($k = 2)
		{
			$this->selectMeans($k);

			while (TRUE)
			{
				$this->log('Iteration #'.$this->iter);

				$this->cdata = array_fill(0, $k, array());

				foreach ($this->data as $ix => $d)
				{
					$dists = array_map(
							function($m) use($d)
							{
								return array_sum(array_map(
										function($a, $b)
										{
											return pow($a - $b, 2);
										}, $d, $m));
							}, $this->means);

					$c = 0;
					$cdist = $dists[0];

					foreach ($dists as $i => $dist)
					{
						if ($dist < $cdist)
						{
							$c = $i;
							$cdist = $dist;
						}
					}

					$this->c[$ix] = $c;
					$this->cdata[$c][] = $d;
				}

				$this->cdata = array_map(
						function($x)
						{
							return array_map(
									function($x)
									{
										return Statistics::mean($x);
									}, MatrixOps::transpose($x));
						}, $this->cdata);

				$this->log('New means:');
				$this->log(implode("\n", array_map(
						function($m)
						{
							return '('.implode(', ', $m).')';
						}, $this->cdata)));

				if (! array_reduce(array_map(
						function($a, $b)
						{
							return array_reduce(array_map(
									function($a, $b)
									{
										return $a != $b;
									}, $a, $b),
									function($a, $b)
									{
										return $a || $b;
									}, FALSE);
						}, $this->means, $this->cdata),
						function($a, $b)
						{
							return $a || $b;
						}, FALSE))
				{
					break;
				}

				$this->means = $this->cdata;

				++$this->iter;
			}

			$this->data = array_map(
					function($d, $c)
					{
						return array_merge($d, array($c));
					}, $this->data, $this->c);

			$this->log('Resulting clustering:');
			$this->log(implode("\n", array_map(
					function($x)
					{
						return '('.implode(', ', $x).')';
					}, $this->data)));

			return $this->data;
		}

		final private function __construct()
		{
			$this->iter = 1;
		}

		final private function selectMeans($k)
		{
			$is = array_rand($this->data, $k);

			$data = $this->data;

			$this->means = array_map(
					function($i) use($data)
					{
						return $data[$i];
					}, $is);

			return $this;
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
