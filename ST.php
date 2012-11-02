<?php

	final class ST
	{
		final static public function probs(array $x, $ix = 0)
		{
			$cnt = 0;
			$grp = array();

			foreach ($x as $x0)
			{
				++$cnt;

				if (! array_key_exists($x0[$ix], $grp))
				{
					$grp[$x0[$ix]] = 0;
				}

				++$grp[$x0[$ix]];
			}

			$grp = array_map(
					function($x) use($cnt)
					{
						return $x / $cnt;
					}, $grp);

			return $grp;
		}

		final static public function slice(array $x, array $ix)
		{
			return array_map(
					function($x) use($ix)
					{
						return array_map(
								function($ix) use($x)
								{
									return $x[$ix];
								}, $ix);
					}, $x);
		}

		final static public function dice(array $x, $ix)
		{
			$cnt = 0;
			$grp = array();

			foreach ($x as $x0)
			{
				++$cnt;

				if (! array_key_exists($x0[$ix], $grp))
				{
					$grp[$x0[$ix]] = array(0, array());
				}

				++$grp[$x0[$ix]][0];
				$grp[$x0[$ix]][1][] = $x0;
			}

			$grp = array_map(
					function($x) use($cnt)
					{
						return array($x[0] / $cnt, $x[1]);
					}, $grp);

			return $grp;
		}

		final static public function dicev(array $x, $ix, $xt)
		{
			$cnt = 0;
			$grp = array();

			foreach ($x as $x0)
			{
				$ixr = ($x0[$ix] < $xt) ? 0 : 1;

				++$cnt;

				if (! array_key_exists($ixr, $grp))
				{
					$grp[$ixr] = array(0, array());
				}

				++$grp[$ixr][0];
				$grp[$ixr][1][] = $x0;
			}

			$grp = array_map(
					function($x) use($cnt)
					{
						return array($x[0] / $cnt, $x[1]);
					}, $grp);

			return $grp;
		}

		final static public function H(array $x, $ix = 0)
		{
			$grp = ST::probs($x, $ix);

			$h = 0;

			foreach ($grp as $p)
			{
				$h -= $p * (log($p) / log(2));
			}

			return $h;
		}

		final static public function Hbar(array $x, $ix, $ix0 = 0)
		{
			$grp = ST::dice($x, $ix);

			return array_sum(array_map(
					function($x) use($ix0)
					{
						return $x[0] * ST::H($x[1], $ix0);
					}, $grp));
		}

		final static public function Hstar(array $x, $ix, $x0, $ix0 = 0)
		{
			$grp = ST::dicev($x, $ix, $x0);

			return array_sum(array_map(
					function($x) use($ix0)
					{
						return $x[0] * ST::H($x[1], $ix0);
					}, $grp));
		}

		final static public function IG(array $x, $ix, $ix0 = 0)
		{
			return ST::H($x, $ix0) - ST::Hbar($x, $ix, $ix0);
		}

		final static public function IGstar(array $x, $ix, $ix0 = 0)
		{
			// it might be prohibitively expensive to attempt splits on all real-valued observations
			// consider the alternative of separating the observation range into a fixed number of chunks

			$vals = array_unique(array_map(
					function($x) use($ix)
					{
						return $x[$ix];
					}, $x));

			$ig = 0;
			$v = NULL;

			$h = ST::H($x, $ix0);

			foreach ($vals as $v0)
			{
				$ig0 = $h - ST::Hstar($x, $ix, $v0, $ix0);

				if ($ig0 > $ig)
				{
					$ig = $ig0;
					$v = $v0;
				}
			}

			return array($ig, $v);
		}

		final private function __construct()
		{
		}
	}

?>
