<?php

	/**
	 * Decision tree classifier (without pruning)
	 */
	class Dtree
	{
		private $leaf;
		private $type;
		private $dix;
		private $dv;
		private $decision;
		private $children;

		private $log;

		final static public function make(array $data, array $type, $maxf = 0, $log = NULL)
		{
			return new self($data, $type, $maxf, $log);
		}

		final public function classify(array $data)
		{
			$data0 = $data;

			array_unshift($data0, 0);

			if ($this->leaf)
			{
				return $this->decision;
			}
			elseif ($this->type)
			{
				if (! array_key_exists($data0[$this->dix], $this->children))
				{
					return 0;
				}

				return $this->children[$data0[$this->dix]]->classify($data);
			}
			else
			{
				$ixr = ($data0[$this->dix] < $this->dv) ? 0 : 1;

				return $this->children[$ixr]->classify($data);
			}
		}

		final private function train(array $data, array $type, $maxf = 0)
		{
			$log = $this->log;

			$log(count($data).' observation(s) of '.count($type).' variable(s)');

			$maxf = round($maxf);

			if (($maxf < 1) || ($maxf > (count($type) - 1)))
			{
				$maxf = count($type) - 1;
			}

			$log('H(X0) = '.ST::H($data));

			$maxig = 0;
			$maxigix = 0;
			$maxigv = NULL;

			$ixs = range(1, count($type) - 1);
			shuffle($ixs);
			$ixs = array_merge(array(0), array_slice($ixs, 0, $maxf));

			foreach ($ixs as $ix)
			{
				$t = $type[$ix];

				if ($ix === 0)
				{
					$log('X0: category');

					continue;
				}

				if ($t)
				{
					$ig = ST::IG($data, $ix);
					$v = NULL;

					$log('X'.$ix.': IG(X0 | X'.$ix.') = '.$ig);
				}
				else
				{
					list($ig, $v) = ST::IGstar($data, $ix);

					$log('X'.$ix.': IG(X0 | X'.$ix.') = '.$ig.' (split on '.(is_null($v) ? 'NULL' : $v).')');
				}

				if ($ig > $maxig)
				{
					$maxig = $ig;
					$maxigix = $ix;
					$maxigv = $v;
				}
			}

			if ($maxigix !== 0)
			{
				$log('Splitting on X'.$maxigix.(is_null($maxigv) ? '' : (' < '.$maxigv)).' (IG = '.$maxig.')');

				if ($type[$maxigix])
				{
					$subd = ST::dice($data, $maxigix);

					foreach ($subd as $lab => $x)
					{
						$this->children[$lab] = Dtree::make($x[1], $type, $maxf, $log);
					}
				}
				else
				{
					$subd = ST::dicev($data, $maxigix, $maxigv);

					foreach ($subd as $lab => $x)
					{
						$this->children[$lab] = Dtree::make($x[1], $type, $maxf, $log);
					}
				}

				$this->leaf = FALSE;
				$this->type = $type[$maxigix];
				$this->dix = $maxigix;
				$this->dv = $maxigv;
				$this->decision = NULL;
			}
			else
			{
				$p = ST::probs($data);

				$dp = 0;
				$d = NULL;

				foreach ($p as $d0 => $p0)
				{
					if ($p0 > $dp)
					{
						$dp = $p0;
						$d = $d0;
					}
				}

				$this->leaf = TRUE;
				$this->type = NULL;
				$this->dix = NULL;
				$this->dv = NULL;
				$this->decision = $d0;
				$this->children = NULL;

				$log('Leaf node reached, decision: '.$this->decision);
			}
		}

		final private function __construct(array $data, array $type, $maxf = 0, $log = NULL)
		{
			if (is_callable($log))
			{
				$this->log = $log;
			}
			else
			{
				$this->log = function($x) {};
			}

			$this->train($data, $type, $maxf);
		}
	}

?>
