<?php

	// splits training data into training and cross-validation sets

	shuffle($data);
	$sp = floor(count($data) * 0.75);
	$datacv = array_slice($data, $sp);
	$data = array_slice($data, 0, $sp);

?>
