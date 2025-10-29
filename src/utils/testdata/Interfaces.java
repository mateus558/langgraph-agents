package demo.interfaces;

public interface Computable {
    default int compute(int x) {
        return x * 2;
    }

    static String name() {
        return "Computable";
    }
}
