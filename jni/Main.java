public class Main {
	public native int square(int i);
	public static void main(String[] args) {
		System.loadLibrary("Main");
		System.out.println(new Main().square(2));
	}
}
