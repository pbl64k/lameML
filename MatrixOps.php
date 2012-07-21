<?php

	/**
	 * Simple operations on matrices representes as arrays of arrays
	 */
	final class MatrixOps
	{
		/**
		 * Returns transposition of a given matrix.
		 */
		final static public function transpose(array $x)
		{
			$res = array_fill(0, count($x[0]), array());

			foreach ($x as $line)
			{
				foreach ($line as $i => $elt)
				{
					$res[$i][] = $elt;
				}
			}

			return $res;
		}
	}

?>
