<?php

	class KdTree
	{
		private $dim;
		private $root = NULL;

		final static public function make($dim)
		{
			return new self($dim);
		}

		final public function add(array $pt, $data)
		{
			$this->root = $this->add0($pt, $data, $this->root, 0);

			return $this;
		}

		final public function findKnn(array $pt, $k)
		{
			return $this->findKnn0($pt, $k, $this->root, 0, array());
		}

		final private function __construct($dim)
		{
			$this->dim = intval($dim);
			$this->root = NULL;
		}

		final private function add0(array $pt, $data, $node, $ix)
		{
			if (is_null($node))
			{
				return KdTreeNode::make($pt, $data);
			}

			$pt0 = $node->getPt();

			if ($pt[$ix] < $pt0[$ix])
			{
				$node->setLc($this->add0($pt, $data, $node->getLc(), ($ix + 1) % $this->dim));
			}
			else
			{
				$node->setRc($this->add0($pt, $data, $node->getRc(), ($ix + 1) % $this->dim));
			}

			return $node;
		}

		final private function findKnn0(array $pt, $k, $node, $ix, array $res)
		{
			if (is_null($node))
			{
				return $res;
			}

			$res = $this->injectNn($pt, $node->getPt(), $k, $node->getData(), $res);

			$pt0 = $node->getPt();

			if ($pt[$ix] < $pt0[$ix])
			{
				$res = $this->findKnn0($pt, $k, $node->getLc(), ($ix + 1) % $this->dim, $res);

				list($d, $p, $dt) = $res[count($res) - 1];

				//if ((count($res) < $k) || ($d > abs($pt0[$ix] - $pt[$ix])))
				{
					$res = $this->findKnn0($pt, $k, $node->getRc(), ($ix + 1) % $this->dim, $res);
				}
			}
			else
			{
				$res = $this->findKnn0($pt, $k, $node->getRc(), ($ix + 1) % $this->dim, $res);

				list($d, $p, $dt) = $res[count($res) - 1];

				//if ((count($res) < $k) || ($d > abs($pt0[$ix] - $pt[$ix])))
				{
					$res = $this->findKnn0($pt, $k, $node->getLc(), ($ix + 1) % $this->dim, $res);
				}
			}

			return $res;
		}

		final private function injectNn(array $pt0, array $pt, $k, $data, array $res0)
		{
			$dist = $this->calcDist($pt0, $pt);

			if (! count($res0))
			{
				return array(array($dist, $pt, $data));
			}

			$res = array();

			$injected = FALSE;

			$rsc = count($res0);
			$rc = min($k, $rsc + 1);

			for ($i = 0; $i != $rc; ++$i)
			{
				if (! $injected)
				{
					if ($i === $rsc)
					{
						$res[] = array($dist, $pt, $data);

						$injected = TRUE;

						continue;
					}

					list($d, $p, $dt) = $res0[$i - ($injected ? 1 : 0)];

					if ($dist < $d)
					{
						$res[] = array($dist, $pt, $data);

						$injected = TRUE;

						continue;
					}
				}

				$res[] = $res0[$i - ($injected ? 1 : 0)];
			}

			return $res;
		}

		final private function calcDist(array $pt0, array $pt)
		{
			return sqrt(array_sum(array_map(
					function($a, $b)
					{
						$c = $a - $b;
						return $c * $c;
					}, $pt0, $pt)));
		}
	}

?>
