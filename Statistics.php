<?php

	/* Some simple statistical functions */
	final class Statistics
	{
		/**
		 * Computes mean of data
		 */
		final static public function mean(array $x)
		{
			return array_sum($x) / count($x);
		}

		/**
		 * Computes variance of data (square of standard deviation)
		 */
		final static public function variance(array $x)
		{
			$m = self::mean($x);

			$z = array_map(function($x) use($m) { return ($x - $m) * ($x - $m); }, $x);

			return self::mean($z);
		}

		/**
		 * Computes standard deviation of data
		 */
		final static public function stddev(array $x)
		{
			return sqrt(self::variance($x));
		}

		/**
		 * Normalize given data as (data - mean) / stddev
		 */
		final static public function normalize(array $x)
		{
			$m = self::mean($x);
			$s = self::stddev($x);

			if ($s == 0)
			{
				$s = 1;
			}

			return array($m, $s,
					array_map(
							function($x) use($m, $s)
							{
								return ($x - $m) / $s;
							}, $x));
		}
	}

?>
