#!/usr/bin/env perl
use KNP;
use strict;

my $knp = new KNP;
while (<STDIN>) {
	my $result = $knp->parse($_);
	if ($result) {
		for my $mrph ($result->mrph) {
			print $mrph->genkei,"\n";
			print "原型: ",$mrph->genkei,"\n";
			print "読み: ",$mrph->yomi,"\n";
			print "品詞: ",$mrph->hinsi,"\n";
			print "活用1: ",$mrph->katuyou1,"\n";
			print "活用2: ",$mrph->katuyou2,"\n";
		}
	}
}
