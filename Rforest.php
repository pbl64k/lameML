<?php

	/**
	 * Ensemble of decision trees trained from bootstrapped data
	 */
	class Rforest
	{
		private $forest;

		private $log;

		final static public function make(array $data, array $type, $n = 50, $log = NULL)
		{
			return new self($data, $type, $n, $log);
		}

		final public function classify(array $data)
		{
			$vote = array();

			foreach ($this->forest as $tree)
			{
				$res = $tree->classify($data);

				if (! array_key_exists($res, $vote))
				{
					$vote[$res] = 0;
				}

				++$vote[$res];
			}

			$maxv = 0;
			$maxd = NULL;

			foreach ($vote as $d => $v)
			{
				if ($v > $maxv)
				{
					$maxv = $v;
					$maxd = $d;
				}
			}

			return $maxd;
		}

		final private function train(array $data, array $type, $n = 50)
		{
			$log = $this->log;

			$m = count($type) - 1;
			$nn = count($data);

			for ($i = 0; $i != $n; ++$i)
			{
				$log('Training tree No. '.($i + 1).'...');

				$log('Resampling...');

				$data0 = array();

				for ($j = 0; $j != $nn; ++$j)
				{
					$data0[] = $data[rand(0, $nn - 1)];
				}

				$log('Training...');

				//$this->forest[] = Dtree::make($data0, $type, round(sqrt($m)) + 1, $log);
				$this->forest[] = Dtree::make($data0, $type, round(sqrt($m)), $log);

				$log('Done!');
			}

			$log('All done.');
		}

		final private function __construct(array $data, array $type, $n = 50, $log = NULL)
		{
			if (is_callable($log))
			{
				$this->log = $log;
			}
			else
			{
				$this->log = function($x) {};
			}

			$this->train($data, $type, $n);
		}
	}

?>
