package test;

import java.io.*;

public class TestRuntime {

	public static void main(String[] args) {
		BufferedReader br_in = new BufferedReader(new InputStreamReader(System.in));
		BufferedReader br    = null;
		String s = null;
		try {
			while ((s = br_in.readLine()) != null) {
				System.out.println(s);
				if (s.equals("")) { break; }
				String [] command = {"/bin/bash", "-c", String.format("echo %s | /usr/bin/env perl script/TestKNP_mrph.pl", new Object[] {s})};
				Process process = Runtime.getRuntime().exec(command);
				InputStream is = process.getInputStream();
				br = new BufferedReader(new InputStreamReader(is));
				String line;
				while ((line = br.readLine()) != null) {
					System.out.println(line);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			if (br    != null) try { br   .close(); } catch (Exception e) {}
			if (br_in != null) try { br_in.close(); } catch (Exception e) {}
		}
	}
}
