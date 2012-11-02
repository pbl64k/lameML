<?php

	class KdTreeNode
	{
		private $pt;
		private $data;
		private $lc = NULL;
		private $rc = NULL;

		final static public function make(array $pt, $data)
		{
			return new self($pt, $data);
		}

		final public function getPt()
		{
			return $this->pt;
		}

		final public function getData()
		{
			return $this->data;
		}

		final public function setLc($lc)
		{
			$this->lc = $lc;

			return $this;
		}

		final public function getLc()
		{
			return $this->lc;
		}

		final public function setRc($rc)
		{
			$this->rc = $rc;

			return $this;
		}

		final public function getRc()
		{
			return $this->rc;
		}

		final private function __construct(array $pt, $data)
		{
			$this->pt = $pt;
			$this->data = $data;
			$this->lc = NULL;
			$this->rc = NULL;
		}
	}

?>
